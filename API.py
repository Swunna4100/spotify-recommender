import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import streamlit as st


# ------------------- DATA LOADING & CLEANING -------------------
def load_data():
    sd = pd.read_csv("spotifydata.csv")

    # keeping the columns we need
    sd = sd[[
        "track_name", "track_artist", "track_popularity", "playlist_genre",
        "danceability", "energy", "tempo", "valence", "acousticness", "instrumentalness"
    ]]
    sd.drop_duplicates(inplace=True)

    # scaling these numbers so they are all between 0 and 1
    num_columns = ["danceability", "energy", "tempo", "valence", "acousticness", "instrumentalness", "track_popularity"]
    scaler = MinMaxScaler()
    sd[num_columns] = scaler.fit_transform(sd[num_columns])

    # fixing songs mapped to wrong genre
    afrobeats_tracks = [
        "OZEBA", "Fi Kan We Kan", "Kese (Dance)", "alone - Remix", "JUJU (feat. Shallipopi)"
    ]
    sd.loc[sd['track_name'].isin(afrobeats_tracks), 'playlist_genre'] = 'afrobeats'
    sd.loc[sd['playlist_genre'] == 'arabic', 'playlist_genre'] = 'hip-hop'

    return sd, num_columns


# ------------------- SONG SEARCH FUNCTION -------------------
def find_song_match(song_name, sd):
    """finds the closest matching song"""
    all_songs = sd['track_name'].tolist()
    closest = difflib.get_close_matches(song_name, all_songs, n=1, cutoff=0.35)

    if not closest:
        return None

    matched_song = closest[0]
    user_song = sd[sd['track_name'] == matched_song].iloc[0]
    return user_song


# ------------------- RECOMMENDER FUNCTION -------------------
def recommend_songs(user_song, sd, num_columns, top_n=5):
    """generates recommendations based on the confirmed song"""
    # getting the audio features
    user_features = user_song[num_columns].values.reshape(1, -1)
    sd_features = sd[num_columns].values
    song_sim = cosine_similarity(user_features, sd_features)[0]

    # checking how similar the genres are
    genre_sim = (sd['playlist_genre'] == user_song['playlist_genre']).astype(float)

    # mixing both similarities together (60% genre, 40% audio features)
    combined_similarity = (0.6 * genre_sim) + (0.4 * song_sim)
    sd['combined_similarity'] = combined_similarity

    # making sure we don't recommend the same song back to them
    recommendations = sd[sd['track_name'] != user_song['track_name']]
    recommendations = recommendations.sort_values(by='combined_similarity', ascending=False)

    # the top recommendations
    top_recs = recommendations[['track_name', 'track_artist', 'playlist_genre', 'combined_similarity']].head(top_n)
    return top_recs


# ------------------- STREAMLIT APP -------------------
st.title("🎶 Spotify Song Recommender")

# load up the data
sd, num_columns = load_data()

# variables that keep track of events
if 'matched_song' not in st.session_state:
    st.session_state.matched_song = None
if 'show_confirmation' not in st.session_state:
    st.session_state.show_confirmation = False
if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False

# Input field
song_choice = st.text_input("Enter a song you like:")

# Search button
if st.button("Search Song"):
    if song_choice.strip() != "":
        matched_song = find_song_match(song_choice, sd)
        if matched_song is None:
            st.error("❌ No close matches found. Try another song.")
            st.session_state.show_confirmation = False
            st.session_state.confirmed = False
        else:
            st.session_state.matched_song = matched_song
            st.session_state.show_confirmation = True
            st.session_state.confirmed = False
    else:
        st.warning("⚠️ Please type a song name.")

# Show confirmation if a match was found
if st.session_state.show_confirmation and st.session_state.matched_song is not None:
    matched_song = st.session_state.matched_song

    st.info(f"🎵 Did you mean: **{matched_song['track_name']}** by **{matched_song['track_artist']}**?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes, that's correct!"):
            st.session_state.confirmed = True
            st.session_state.show_confirmation = False

    with col2:
        if st.button("❌ No, search again"):
            st.session_state.matched_song = None
            st.session_state.show_confirmation = False
            st.session_state.confirmed = False
            st.rerun()

# Show recommendations if confirmed
if st.session_state.confirmed and st.session_state.matched_song is not None:
    user_song = st.session_state.matched_song

    st.success(f" You picked: **{user_song['track_name']}** by **{user_song['track_artist']}**")

    # Get recommendations
    with st.spinner("Finding similar songs..."):
        recommendations = recommend_songs(user_song, sd, num_columns)

    st.subheader(" Here are your recommendations:")
    st.dataframe(recommendations, use_container_width=True)

    # Reset button
    if st.button("🔄 Search for another song"):
        st.session_state.matched_song = None
        st.session_state.show_confirmation = False
        st.session_state.confirmed = False
        st.rerun()