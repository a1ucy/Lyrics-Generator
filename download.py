import lyricsgenius
import pandas as pd

genius = lyricsgenius.Genius(access_token='E1x3pJL8KZsntid7CZNr9cgU1tlkD375h5KUOfwijD3dtUJgMtVCT7B7qfUEpk9B', timeout=40, sleep_time=2)

artist = genius.search_artist(artist_name='Drake',max_songs=100)
# print(artist.songs)
lyrics = []
for i in artist.songs:
    lyrics.append(i.lyrics)

print('len:',len(lyrics))

data = pd.DataFrame(lyrics)
data.to_csv('drake_songs.csv',index=False)

print('DONE!!!!!')

