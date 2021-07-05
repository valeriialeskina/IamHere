#importing libraries
from gensim.utils import dict_from_corpus
import pandas as pd
import numpy as np
from nrclex import NRCLex
import streamlit as st
import pickle
import plotly.express as px
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)
from gensim.parsing.preprocessing import STOPWORDS, strip_punctuation, strip_short, strip_punctuation
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from functions import clean_data, get_emotion_nrclx, get_emotion_scores, calcu1, get_top_sentences_emotions, _preprocess_text, remove_special_characters
import datetime

#dataset (blog) filter for analysis of emotions
blog = st.sidebar.selectbox(label = 'Select the Blog', options=['Diary Blog', 'Travel Blog', 'Depression Blog'])
if blog == 'Depression Blog':
    df = pickle.load(open('data/depression_marathon_df_final.pkl', 'rb'))
elif blog == 'Diary Blog':
    df = pickle.load(open('data/george_diary_df_final.pkl','rb'))
else:
    df = pickle.load(open('data/travel_blog_df_final.pkl', 'rb'))


df['emotion'] = df.full_text.apply(get_emotion_nrclx) #getting emotions from nrclx library
st.title("IamHere Dashboard")
st.write("")
period_start = st.sidebar.date_input('Choose the start date for entries', value=datetime.date(2021,5,1), min_value=df.date.min(), max_value=df.date.max())
period_end = st.sidebar.date_input('Choose the end date for entries', min_value=df.date.min(), max_value=df.date.max())
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.markdown("**_IamHere_** to make you understand yourself better. I will show the diary entries and emoions dervied from those entries. Emotions can be analyzed on aggregate level (all entries) and entry level (specific diary entry). For analysis of specific entries, scroll down the dashboard")
if period_start > period_end:
    st.error('Error: End date must be after start date.')
#else:
    #st.success('Start date: `%s`\n\nEnd date: `%s`' % (period_start, period_end))



df_subset = df.loc[(df['date'].dt.date >= period_start) & (df['date'].dt.date < period_end)] #subsetting the data based on user-defined ranges
df_subset_grouped = df_subset.groupby([pd.Grouper(key='date', freq='7D'), 'emotion']).size().reset_index(name='count') #group by dates and get counts
st.write("Number of entries across the specified period:", len(df_subset))
table = st.write(df_subset[['header','date', 'full_text']])


col1, col2 = st.beta_columns((1,2))
fig1 = px.bar(data_frame=df_subset_grouped, x = 'date', y = 'count', color = 'emotion', color_discrete_map= 
{'positive':'steelblue', 'negative':'firebrick', 'anticipation':'orange', 'trust':'green',
'fear':'purple'}, title = 'Bar Chart with dynamics of Emotions for Specified timescale')
#fig1.update_yaxes(tickformat='d')
fig2 = px.pie(data_frame=df_subset_grouped, names = 'emotion', color= 'emotion', color_discrete_map=
{'positive':'steelblue', 'negative':'firebrick', 'anticipation':'orange', 'trust':'green',
'fear':'purple'}, title = 'Pie Chart with frequency of emotions')


col1.plotly_chart(fig2, use_container_width=True)
col2.plotly_chart(fig1, use_container_width=True)
emotion = st.selectbox(label = 'Choose emotion to see most common words', options = list(df_subset.emotion.unique()))
df_subset['full_text_clean'] = df_subset.full_text.apply(clean_data).apply(remove_special_characters)
#df_subset['full_text_clean'] = df_subset.full_text.apply(clean_data).apply(_preprocess_text)
df_subset_emotions = df_subset.full_text_clean.loc[df_subset.emotion == emotion]
df_subset_emotions_text = ''.join(df_subset_emotions)
stopwords = set(STOPWORDS)
#plt.figure(figsize=(10,8))
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(df_subset_emotions_text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
st.pyplot()


df_subset_emotion = df_subset.loc[df_subset.emotion == emotion]
sent = get_top_sentences_emotions(df_subset_emotion.full_text, emotion_segment=emotion)
st.markdown('**Top 3 sentences of selected emotion from your diary**', unsafe_allow_html=False)
st.write('')
for i in sent[:3]:
    st.write(i)

st.write('')
st.write('')

##entry-level filtering section
entry_level = st.checkbox(label='Select this to go to the analysis of specific diary entry')
if entry_level is True:
    entry_diary = st.selectbox(label='Choose the diary entry', options = list(df_subset.header.unique()))
    df_entry = df.full_text[df.header == entry_diary]
    st.subheader('Diary text')
    st.write(df_entry.iloc[0][:500],'...')
    col1, col2 = st.beta_columns(2)
    with col1:
        fig3 = px.pie(data_frame=df_subset_grouped, names = 'emotion', color= 'emotion', color_discrete_map=
        {'positive':'steelblue', 'negative':'firebrick', 'anticipation':'orange', 'trust':'green',
        'fear':'purple'}, title = 'Frequency of emotions in the selected entry')
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        st.subheader('Top words used in the text')
        wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(df_entry.iloc[0])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()
        st.pyplot()  
    st.markdown('**Top 3 sentences of selected emotion from your specified diary entry**')
    sent2 = get_top_sentences_emotions(df_entry, emotion)
    for i in sent2:
        st.write(i)
 
st.write('')
st.write('')
if  st.checkbox("Get REAL-TIME emotions! Check this mark and try it out yourself!"):
    
    abc = df_subset['full_text'].iloc[0][:500]
    message = st.text_area("Just write your thoughts", abc)
    if st.button("Submit"):
         col1,col2 =st.beta_columns(2)
         with col1:
           st.subheader('Emotions in your entry')
           blob = calcu1(message)
           data_items = blob.items()
           data_list = list(data_items)
           df = pd.DataFrame(data_list,columns = ['Emotion','value'])
           df1= df[df['value'] != 0]
           a_list = df1['Emotion'].tolist()
           fig = px.pie(df1, values= df1.value,names= a_list)
           st.plotly_chart(fig,use_container_width=True)
         with col2:
           st.subheader('Top words used in the text')
           wordcloud2 = WordCloud(background_color='white',width=400, height=200).generate(message)
           st.set_option('deprecation.showPyplotGlobalUse', False)
           plt.imshow(wordcloud2)
           plt.axis("off")
           st.pyplot()
