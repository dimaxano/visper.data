import os

from pytube import exceptions as pytube_exceptions, YouTube as YTDownloader
from .utils import video_id_to_url
from .youtube_api import YoutubeApi
from tqdm import tqdm

resolutions_list = ['1080p', '720p', '360p', '240p', '144p']


def download_video(video_url: str, resolution='1080p', save_path='./video_downloads'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    try:
        yt = YTDownloader(video_url)
        streams = yt.streams.filter(resolution=resolution, progressive=True)
        if len(streams.fmt_streams) == 0:
            for r in resolutions_list:
                streams = yt.streams.filter(resolution=r, progressive=True)
                if len(streams.fmt_streams) > 0:
                    break
        stream = streams.first()
        stream.download(save_path)
    except Exception:
        print("{} can't be downloaded".format(video_url))


def download_channel_videos(channel_id: str, yt_api=None, yt_api_key='', save_path='./video_downloads', verbose=False):
    if not yt_api:
        if yt_api_key:
            yt_api = YoutubeApi(yt_api_key)
        else:
            print('No youtube api key provided')
            return
    videos_ids = yt_api.get_channel_videos_ids(channel_id)
    if verbose:
        videos_ids = tqdm(videos_ids)
    for v_id in videos_ids:
        download_video(video_id_to_url(v_id), save_path=save_path)


def download_playlist_videos(playlist_id: str, yt_api=None, yt_api_key='', save_path='./video_downloads', verbose=False):
    if not yt_api:
        if yt_api_key:
            yt_api = YoutubeApi(yt_api_key)
        else:
            print('No youtube api key provided')
            return
    videos_ids = yt_api.get_playlist_videos_ids(playlist_id)
    if verbose:
        videos_ids = tqdm(videos_ids)
    for v_id in videos_ids:
        download_video(video_id_to_url(v_id), save_path=save_path)


