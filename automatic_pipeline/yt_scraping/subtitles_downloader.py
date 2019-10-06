from youtube_transcript_api import YouTubeTranscriptApi


def get_subtitles(video_ids):
    return YouTubeTranscriptApi.get_transcripts(video_ids)


if __name__ == '__main__':
    vid_ids = ['ty6BuTTYD4w']
    print(get_subtitles(vid_ids))
