CHANNEL_PREFIX = 'https://www.youtube.com/channel/'
PLAYLIST_PREFIX = 'https://www.youtube.com/playlist?list='
VIDEO_PREFIX = 'https://www.youtube.com/watch?v='


def channel_url_to_id(channel_url: str):
    return channel_url.replace(CHANNEL_PREFIX, '')


def channel_id_to_url(channel_id: str):
    return '{}{}'.format(CHANNEL_PREFIX, channel_id)


def playlist_url_to_id(playlist_url: str):
    return playlist_url.replace(PLAYLIST_PREFIX, '')


def playlist_id_to_url(playlist_id: str):
    return '{}{}'.format(PLAYLIST_PREFIX, playlist_id)


def video_url_to_id(video_url: str):
    return video_url.replace(VIDEO_PREFIX, '')


def video_id_to_url(video_id: str):
    return '{}{}'.format(VIDEO_PREFIX, video_id)