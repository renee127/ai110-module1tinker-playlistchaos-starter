"""Core playlist logic: normalize songs, classify moods, compute stats, pick songs."""

import random
from typing import Any, Dict, Final, FrozenSet, List, Mapping, Optional, Tuple

Song = Dict[str, Any]
Profile = Mapping[str, Any]
PlaylistMap = Dict[str, List[Song]]

# --- Mood labels -------------------------------------------------------------

HYPE: Final = "Hype"
CHILL: Final = "Chill"
MIXED: Final = "Mixed"
MOODS: Final[Tuple[str, ...]] = (HYPE, CHILL, MIXED)

# --- Classification rules ----------------------------------------------------

# Matched as substrings of the song's genre.
HYPE_KEYWORDS: Final[FrozenSet[str]] = frozenset({"rock", "punk", "party"})
# Matched as substrings of the song's (lowercased) title.
CHILL_KEYWORDS: Final[FrozenSet[str]] = frozenset({"lofi", "ambient", "sleep"})

DEFAULT_HYPE_MIN_ENERGY: Final = 7
DEFAULT_CHILL_MAX_ENERGY: Final = 3

DEFAULT_PROFILE: Final[Dict[str, Any]] = {
    "name": "Default",
    "hype_min_energy": DEFAULT_HYPE_MIN_ENERGY,
    "chill_max_energy": DEFAULT_CHILL_MAX_ENERGY,
    "favorite_genre": "rock",
    "include_mixed": True,
}


# --- Normalization -----------------------------------------------------------


def normalize_title(title: Any) -> str:
    """Strip surrounding whitespace from a title, keeping its display casing.

    Non-string input (including None) becomes an empty string.
    """
    if not isinstance(title, str):
        return ""
    return title.strip()


def normalize_artist(artist: Any) -> str:
    """Return the artist name stripped and lowercased; empty string if missing."""
    if not artist:
        return ""
    return str(artist).strip().lower()


def normalize_genre(genre: Any) -> str:
    """Return the genre stripped and lowercased; empty string if missing."""
    if not genre:
        return ""
    return str(genre).strip().lower()


def _to_int(value: Any, default: int = 0) -> int:
    """Coerce an int, float, or numeric string (e.g. " 7.5 ") to int.

    Returns `default` when the value cannot be parsed.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return default
    return default


def _normalize_tags(tags: Any) -> List[str]:
    """Return tags as a list of stripped, lowercased, non-empty strings.

    Accepts a single string, an iterable of values, or None.
    """
    if not tags:
        return []
    if isinstance(tags, str):
        tags = [tags]
    cleaned = (str(tag).strip().lower() for tag in tags)
    return [tag for tag in cleaned if tag]


def normalize_song(raw: Mapping[str, Any]) -> Song:
    """Return a clean song dict with title, artist, genre, energy, and tags.

    Missing or None fields fall back to empty values; energy falls back to 0.
    Does not classify the song; see `classify_song`.
    """
    return {
        "title": normalize_title(raw.get("title")),
        "artist": normalize_artist(raw.get("artist")),
        "genre": normalize_genre(raw.get("genre")),
        "energy": _to_int(raw.get("energy", 0)),
        "tags": _normalize_tags(raw.get("tags")),
    }


# --- Classification ----------------------------------------------------------


def _is_hype(energy: int, genre: str, favorite_genre: str, min_energy: int) -> bool:
    """True if the song meets any Hype rule: favorite genre, energy, or keyword."""
    if favorite_genre and genre == favorite_genre:
        return True
    if energy >= min_energy:
        return True
    return any(keyword in genre for keyword in HYPE_KEYWORDS)


def _is_chill(energy: int, title: str, max_energy: int) -> bool:
    """True if the song meets any Chill rule: low energy or a title keyword."""
    if energy <= max_energy:
        return True
    return any(keyword in title for keyword in CHILL_KEYWORDS)


def classify_song(song: Mapping[str, Any], profile: Profile) -> str:
    """Return the mood label ("Hype", "Chill", or "Mixed") for a normalized song.

    Hype rules are checked first, then Chill; anything else is Mixed.
    Thresholds and favorite genre come from `profile`.
    """
    energy = _to_int(song.get("energy", 0))
    genre = normalize_genre(song.get("genre"))
    title = normalize_title(song.get("title")).lower()

    favorite_genre = normalize_genre(profile.get("favorite_genre"))
    hype_min = _to_int(profile.get("hype_min_energy"), DEFAULT_HYPE_MIN_ENERGY)
    chill_max = _to_int(profile.get("chill_max_energy"), DEFAULT_CHILL_MAX_ENERGY)

    if _is_hype(energy, genre, favorite_genre, hype_min):
        return HYPE
    if _is_chill(energy, title, chill_max):
        return CHILL
    return MIXED


# --- Playlist building -------------------------------------------------------


def _empty_playlists() -> PlaylistMap:
    """Return a playlist map with an empty list for every mood."""
    return {mood: [] for mood in MOODS}


def build_playlists(songs: List[Mapping[str, Any]], profile: Profile) -> PlaylistMap:
    """Normalize, classify, and group raw songs into mood playlists.

    Each returned song carries a "mood" key. Input order is preserved.
    """
    playlists = _empty_playlists()
    for raw in songs:
        song = normalize_song(raw)
        song["mood"] = classify_song(song, profile)
        playlists[song["mood"]].append(song)
    return playlists


def merge_playlists(a: PlaylistMap, b: PlaylistMap) -> PlaylistMap:
    """Return a new map with each mood's songs from `a` followed by `b`.

    Neither input is modified.
    """
    merged: PlaylistMap = {}
    for key in list(a) + [k for k in b if k not in a]:
        merged[key] = [*a.get(key, []), *b.get(key, [])]
    return merged


# --- Stats -------------------------------------------------------------------


def _song_key(song: Mapping[str, Any]) -> Tuple[str, str]:
    """Identity key for de-duplication: case-insensitive (title, artist)."""
    return (
        normalize_title(song.get("title")).lower(),
        normalize_artist(song.get("artist")),
    )


def unique_songs(songs: List[Song]) -> List[Song]:
    """Return songs with duplicates removed, keeping the first occurrence."""
    seen: set = set()
    result: List[Song] = []
    for song in songs:
        key = _song_key(song)
        if key in seen:
            continue
        seen.add(key)
        result.append(song)
    return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 when the denominator is zero."""
    if not denominator:
        return 0.0
    return numerator / denominator


def compute_playlist_stats(playlists: PlaylistMap) -> Dict[str, Any]:
    """Compute counts, hype ratio, average energy, and top artist.

    All figures count each unique song (by title and artist) once.
    Empty input yields zeroed stats rather than raising.
    """
    hype = unique_songs(playlists.get(HYPE, []))
    chill = unique_songs(playlists.get(CHILL, []))
    mixed = unique_songs(playlists.get(MIXED, []))
    all_songs = unique_songs([s for songs in playlists.values() for s in songs])

    stats: Dict[str, Any] = {
        "total_songs": len(all_songs),
        "hype_count": len(hype),
        "chill_count": len(chill),
        "mixed_count": len(mixed),
        "hype_ratio": 0.0,
        "avg_energy": 0.0,
        "top_artist": "",
        "top_artist_count": 0,
    }
    if not all_songs:
        return stats

    total_energy = sum(_to_int(song.get("energy", 0)) for song in all_songs)
    top_artist, top_count = most_common_artist(all_songs)

    stats["hype_ratio"] = _safe_ratio(len(hype), len(all_songs))
    stats["avg_energy"] = _safe_ratio(total_energy, len(all_songs))
    stats["top_artist"] = top_artist
    stats["top_artist_count"] = top_count
    return stats


def most_common_artist(songs: List[Song]) -> Tuple[str, int]:
    """Return (artist, count) for the most frequent artist, or ("", 0) if none.

    Ties go to the artist seen first.
    """
    counts: Dict[str, int] = {}
    for song in songs:
        artist = str(song.get("artist") or "")
        if artist:
            counts[artist] = counts.get(artist, 0) + 1

    if not counts:
        return "", 0
    artist = max(counts, key=counts.__getitem__)
    return artist, counts[artist]


# --- Search & picking --------------------------------------------------------


def search_songs(
    songs: List[Song],
    query: Optional[str],
    field: str = "artist",
) -> List[Song]:
    """Return songs whose `field` contains `query` (case-insensitive substring).

    An empty or blank query returns all songs unchanged.
    """
    q = (query or "").strip().lower()
    if not q:
        return songs
    return [song for song in songs if q in str(song.get(field) or "").lower()]


def lucky_pick(playlists: PlaylistMap, mode: str = "any") -> Optional[Song]:
    """Pick a random song according to `mode`.

    "hype" or "chill" (case-insensitive) pick only from that playlist; any other
    value picks from every playlist. Returns None if there is nothing to pick.
    """
    mode = str(mode).strip().lower()
    if mode == HYPE.lower():
        candidates = playlists.get(HYPE, [])
    elif mode == CHILL.lower():
        candidates = playlists.get(CHILL, [])
    else:
        candidates = [song for songs in playlists.values() for song in songs]
    return random_choice_or_none(candidates)


def random_choice_or_none(songs: List[Song]) -> Optional[Song]:
    """Return a random song from `songs`, or None if the list is empty."""
    if not songs:
        return None
    return random.choice(songs)


# --- History -----------------------------------------------------------------


def history_summary(history: List[Song]) -> Dict[str, int]:
    """Count picked songs per mood; unknown or missing moods count as Mixed."""
    counts = {mood: 0 for mood in MOODS}
    for song in history:
        mood = song.get("mood", MIXED)
        counts[mood if mood in counts else MIXED] += 1
    return counts
