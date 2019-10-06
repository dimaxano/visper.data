from .youtube_downloader import download_channel_videos, download_playlist_videos, download_video
from .youtube_api import YoutubeApi

YOUTUBE_API_KEY = ''
test_channels = []
test_playlists = ['PLUW3M4xT-Qm97ghuWbOec5DGNC12IloPv']
test_videos = ['https://www.youtube.com/watch?v=668nUCeBHyY']

if __name__ == '__main__':
    try:
        yt_api = YoutubeApi(YOUTUBE_API_KEY)
    except ValueError as e:
        print(e)
        exit(-1)
    print('Downloading from channels')
    print('Number of channels to download: ' + str(len(test_channels)))
    for channel_id in test_channels:
        download_channel_videos(channel_id=channel_id, yt_api=yt_api,
                                save_path='./yt_scraping/video_downloads/channels/{}'.format(channel_id),
                                verbose=True)

    print('Downloading from playlists')
    print('Number of playlists to download: ' + str(len(test_playlists)))
    for playlist_id in test_playlists:
        download_playlist_videos(playlist_id=playlist_id,
                                 yt_api=yt_api,
                                 save_path='./yt_scraping/video_downloads/playlists/{}'.format(playlist_id),
                                 verbose=True)
    print('Downloading videos fom links')
    print('Number of videos to download: ' + str(len(test_videos)))
    for video_url in test_videos:
        download_video(video_url=video_url, save_path='./yt_scraping/video_downloads/videos')
