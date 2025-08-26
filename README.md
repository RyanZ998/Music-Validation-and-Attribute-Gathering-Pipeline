# Music-Validation-and-Attribute-Gathering-Pipeline

A pipeline for processing and analyzing music data, including audio features, lyrics sentiment, and more.

## Project Structure
- `apply_rubric.py`: Apply rubric to processed data
- `check_step1.py`: Check results of step 1
- `quick_diag_known_ids.py`: Diagnostics for known song IDs
- `songs.csv`: Main song data
- `step1_audio_features_failed.txt`: Log of failed audio feature extraction
- `step1_enrich.py`: Enrich data in step 1
- `step1_not_found.txt`: Log of not found items in step 1
- `step1_search_hits.csv`: Search hits from step 1
- `step1_spotify_features.py`: Spotify audio feature extraction
- `step1_spotify_output.csv`: Output from Spotify features
- `step2_lyrics_output.csv`: Lyrics output from step 2
- `step2_lyrics_sentiment.py`: Sentiment analysis for lyrics
- `step3_gpt_fill.py`: GPT-based data filling
- `step4_merge_back.py`: Merge results back into main data

## Getting Started
1. Clone this repository
2. Install dependencies (if any)
3. Run the scripts as needed

## License
MIT License

## Tools
This project uses data from [GetSongBPM](https://getsongbpm.com).

