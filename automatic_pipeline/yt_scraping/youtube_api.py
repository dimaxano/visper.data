import json
import requests
import urllib


class YoutubeApi:
    """
    Class wrapping Youtube API.
    Allows getting ids for videos from channels or playlists.
    Needs API key for access (look https://developers.google.com/youtube/v3/getting-started)
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError('No api key provided')
        self.api_key = api_key
        self._channel_url_prefix = 'https://www.googleapis.com/youtube/v3/search?'
        self._playlist_url_prefix = 'https://www.googleapis.com/youtube/v3/playlistItems?'

    def get_channel_videos_ids(self, channel_id: str):
        url = '{}{}'.format(self._channel_url_prefix,
                            urllib.parse.urlencode({'key': self.api_key, 
                                                    'channelId': channel_id,
                                                    'part': 'snippet,id',
                                                    'maxResults': 50}))
        first_url = url

        videos = []
        while True:
            try:
                inp = requests.get(url)
            except requests.exceptions.BaseHTTPError:
                print('Bad url %s', url)
                break

            resp = json.loads(inp.text)

            if 'items' not in resp:
                print('No videos for channel ' + channel_id)
                break

            for i in resp['items']:
                if i['id']['kind'] == "youtube#video":
                    videos.append(i['id']['videoId'])
            try:
                next_page_token = resp['nextPageToken']
                url = first_url + '&pageToken={}'.format(next_page_token)
            except KeyError:
                break
        return videos
        
    def get_playlist_videos_ids(self, playlist_id: str):
        url = '{}{}'.format(self._playlist_url_prefix,
                        urllib.parse.urlencode({'key': self.api_key, 
                                                'playlistId': playlist_id,
                                                'part': 'snippet,contentDetails',
                                                'maxResults': 50}))
        first_url = url

        videos = []
        while True:
            try:
                inp = requests.get(url)
            except requests.exceptions.BaseHTTPError:
                print('Bad url ' + url)
                break

            resp = json.loads(inp.text)
            if 'error' in resp:
                print('API access error')
                break
            if 'items' not in resp:
                print('No videos for playlist ' + playlist_id)
                break

            for i in resp['items']:
                if i['kind'] == "youtube#playlistItem":
                    videos.append(i['contentDetails']['videoId'])
            try:
                next_page_token = resp['nextPageToken']
                url = first_url + '&pageToken={}'.format(next_page_token)
            except KeyError:
                break
        return videos
