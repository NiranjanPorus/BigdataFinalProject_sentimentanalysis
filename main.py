from flask import Flask

from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline(task='sentiment-analysis', model=model, tokenizer=tokenizer)

app=Flask(__name__)

@app.route("/")
def login():
    return "Hi"


@app.route('/intro',methods=['POST','GET'])
def redirect_intro():
    import ktrain
    import os
    import tensorflow as tf
    predicted_sentiment=""
    txt = str(request.form.values())
    get_text(txt)
    if txt is not "":
        model_path = os.path.dirname(os.path.realpath(__file__)) + '/Model_BERT'
        predictor = ktrain.load_predictor(model_path)
        model = ktrain.get_predictor(predictor.model, predictor.preproc)
        predicted_sentiment = model.predict([txt])
    return txt#render_template('intro.html', predicted_sentiment=predicted_sentiment)

def get_text(txt):
    print(f"Text is {txt}")


def tmp():
    import pandas as pd
    df = pd.DataFrame(data={'Name': ['Dom', 'Chris'], 'Sex': ['M', 'M']})
    df.to_html()
    print(df)


def get_tweets():
    import tweepy
    import pandas as pd
    import datetime

    api_key = "dw1HOxVvtFb3YdxyWF7nh53ou"
    api_key_secret = "nVSqVaIHdXbEg63i7VP7zlW3JW0Qd0pHpj484jmOTqX10rVHkv"
    access_token = "1357032758372048896-Js0AuRIqZY0EZVl7diPS1XW8IKyaIl"
    access_token_secret = "lTDpRO6HzwgZHjA2gaXlqh7iBAfMXq3iuTgyO9BGzfYdz"

    hashtags = ['#UkraineRussiaWar', '#StopWarInUkraine', '#UkraineUnderAttack', '#StopPutinNOW', '#UkraineUnderAttack',
                '#RussianWarCrimesInUkraine']

    for i in range(len(hashtags)):
        auth = tweepy.OAuthHandler(api_key, api_key_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)

        # query = tweepy.Cursor(api.search, q=hashtags[i]).items(1)
        query = api.search_tweets('#UkraineRussiaWar', count=2)
        tweets1 = [{'Tweets': tweet.text, 'Timestamp': tweet.created_at, 'lang': tweet.lang, 'Location':tweet.user.location} for tweet in query]  # ,'Location':tweet.location
        # print(tweets1)

        tweet_list = []
        date_list = []
        lang_list = []
        loc_list = []
        for tweet in tweets1:
            txt = tweet.get('Tweets')
            dat = tweet.get('Timestamp').strftime('%m/%d/%Y')
            lang = tweet.get('lang')
            loc=tweet.get('Location')

            tweet_list.append(txt)
            date_list.append(dat)
            lang_list.append(lang)
            loc_list.append(loc)

        df = pd.DataFrame({'Tweet': tweet_list, 'Date': date_list, 'lang': lang_list,'location':loc_list})
        print(df)
    lang_dist = pd.DataFrame({'lang': df.lang.value_counts()})
    return df

def preprocess(df):
    import nltk
    nltk.download("all")
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.corpus import stopwords
    import re
    import contractions
    from textblob import TextBlob

    # df = get_tweets()
    df['Tweet'] = [contractions.fix(str(x)) for x in df.Tweet.values]

    stemmer = WordNetLemmatizer()
    sentences = list(df.Tweet.values)

    for i in range(len(sentences)):
        sentences[i] = re.sub('[@]+ [a-zA-z]*|[@]+[a-zA-z]*', ' ', sentences[i])
        sentences[i] = re.sub('\s*http.*/', ' ', sentences[i], flags=re.MULTILINE)
        sentences[i] = re.sub('www.*html|www.*com', ' ', sentences[i], flags=re.MULTILINE)
        # sentences[i]=re.sub('http.*\/',' ',sentences[i], flags=re.MULTILINE)
        sentences[i] = re.sub('!|,|\.|@|!|\?|\'|\;|%|~|\*|\(|\)|#', '', sentences[i], flags=re.MULTILINE)



        words = nltk.word_tokenize(sentences[i])
        words = [stemmer.lemmatize(x) for x in words if x not in stopwords.words('english')]
        sentences[i] = ' '.join(words)
        # sentences[i]=str(TextBlob(sentences[i]).correct())
        # print(i+1)

        df['Tweet'] = sentences
    # print(df)

    return df


def translate():
    import pandas as pd

    df=get_tweets()
    df=preprocess(df)
    df_non_en = df.loc[df.lang != 'en',]
    df_en = df.loc[df.lang == 'en',]

    lang = pd.DataFrame([['Arabic', 'ar', 'ar_AR'],
                         ['Czech', 'cs', 'cs_CZ'],
                         ['German', 'de', 'de_DE'],
                         ['English', 'en', 'en_XX'],
                         ['Spanish', 'es', 'es_XX'],
                         ['Estonian', 'et', 'et_EE'],
                         ['Finnish', 'fi', 'fi_FI'],
                         ['French', 'fr', 'fr_XX'],
                         ['Gujarati', 'gu', 'gu_IN'],
                         ['Hindi', 'hi', 'hi_IN'],
                         ['Italian', 'it', 'it_IT'],
                         ['Japanese', 'ja', 'ja_XX'],
                         ['Kazakh', 'kk', 'kk_KZ'],
                         ['Korean', 'ko', 'ko_KR'],
                         ['Lithuanian', 'lt', 'lt_LT'],
                         ['Latvian', 'lv', 'lv_LV'],
                         ['Burmese', 'my', 'my_MM'],
                         ['Nepali', 'ne', 'ne_NP'],
                         ['Dutch', 'nl', 'nl_XX'],
                         ['Romanian', 'ro', 'ro_RO'],
                         ['Russian', 'ru', 'ru_RU'],
                         ['Sinhala', 'si', 'si_LK'],
                         ['Turkish', 'tr', 'tr_TR'],
                         ['Vietnamese', 'vi', 'vi_VN'],
                         ['Chinese', 'zh', 'zh_CN'],
                         ['Afrikaans', 'af', 'af_ZA'],
                         ['Azerbaijani', 'az', 'az_AZ'],
                         ['Bengali', 'bn', 'bn_IN'],
                         ['Persian', 'fa', 'fa_IR'],
                         ['Hebrew', 'he', 'he_IL'],
                         ['Croatian', 'hr', 'hr_HR'],
                         ['Indonesian', 'id', 'id_ID'],
                         ['Georgian', 'ka', 'ka_GE'],
                         ['Khmer', 'km', 'km_KH'],
                         ['Macedonian', 'mk', 'mk_MK'],
                         ['Malayalam', 'ml', 'ml_IN'],
                         ['Mongolian', 'mn', 'mn_MN'],
                         ['Marathi', 'mr', 'mr_IN'],
                         ['Polish', 'pl', 'pl_PL'],
                         ['Pashto', 'ps', 'ps_AF'],
                         ['Portuguese', 'pt', 'pt_XX'],
                         ['Swedish', 'sv', 'sv_SE'],
                         ['Swahili', 'sw', 'sw_KE'],
                         ['Tamil', 'ta', 'ta_IN'],
                         ['Telugu', 'te', 'te_IN'],
                         ['Thai', 'th', 'th_TH'],
                         ['Tagalog', 'tl', 'tl_XX'],
                         ['Ukrainian', 'uk', 'uk_UA'],
                         ['Urdu', 'ur', 'ur_PK'],
                         ['Xhosa', 'xh', 'xh_ZA'],
                         ['Galician', 'gl', 'gl_ES'],
                         ['Slovene', 'sl', 'sl_SI']
                         ])
    lang.columns = ['lang', 'lang_code_raw', 'lang_code_bart']

    # Add bart lang codes to df
    list_lang = []
    for i in range(len(df_non_en)):
        if df_non_en.iloc[i]['lang'] in list(lang.lang_code_raw):
            list_lang.extend(list(lang.loc[lang.lang_code_raw == df_non_en.iloc[i]['lang'], 'lang_code_bart']))

    df_non_en.loc[:, 'lang_mbart'] = list_lang

    df_tmp_non_en = df_non_en#.iloc[:5, ]

    from transformers import MBartForConditionalGeneration, MBart50Tokenizer
    model_name = "facebook/mbart-large-50-many-to-many-mmt"

    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)

    list_tweet = []
    for i in range(len(df_tmp_non_en)):
        # print(i)
        tokenizer.src_lang = df_tmp_non_en.iloc[i]['lang_mbart']
        sent = df_tmp_non_en.iloc[i]['Tweet']
        # print(df_non_en.iloc[i]['lang_mbart'])
        # print(sent)
        encoded_non_eng_text = tokenizer(sent, return_tensors="pt")
        generated_tokens = model.generate(**encoded_non_eng_text,
                                          forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
        out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        # print('', str(out).strip('][\''))
        list_tweet.append(str(out).strip('][\''))
        # print(list_tweet)

    df_tmp_non_en['Translated_Tweet'] = list_tweet
    # print(df_tmp_non_en)

    return df_tmp_non_en,df_en

def bert_predict(df_tmp_non_en,df_en):
    df_non_en1 = df_tmp_non_en[['Date', 'Translated_Tweet', 'lang', 'location']]
    df_non_en1.columns = ['Date', 'Tweet', 'lang', 'location']

    import pandas as pd
    df = pd.concat([df_en, df_non_en1], axis=0)

    # import ktrain
    # import os
    # import tensorflow as tf
    # model_path=os.path.dirname(os.path.realpath(__file__)) + '/Model_BERT'


    # predictor = ktrain.load_predictor(model_path)
    # model = ktrain.get_predictor(predictor.model, predictor.preproc)
    # predicted_sentiment = model.predict(list(df.Tweet.values))
    # df['predicted_sentiment']=predicted_sentiment
    # print(df)

    #
    stars = classifier(list(df.Tweet.values))
    predicted_sentiment=[x.get('label') for x in stars]
    df['predicted_sentiment'] = predicted_sentiment
    #
    df.loc[(df.predicted_sentiment == '1 star') | (df.predicted_sentiment == '2 stars'), 'predicted_sentiment'] = 'negative'
    df.loc[(df.predicted_sentiment == '4 stars') | (df.predicted_sentiment == '5 stars'), 'predicted_sentiment'] = 'positive'
    df.loc[df.predicted_sentiment == '3 stars', 'predicted_sentiment'] = 'neutral'
    print(df)
    return df




# def bert_predict(df_tmp_non_en,df_en):
#     df_non_en1 = df_tmp_non_en[['Date', 'Translated_Tweet', 'lang', 'location']]
#     df_non_en1.columns = ['Date', 'Tweet', 'lang', 'location']
# ############
#     import pandas as pd
#     df = pd.concat([df_en, df_non_en1], axis=0)
#     # # import ktrain
#     # # import os
#     # import tensorflow as tf
#     # model_path=os.path.dirname(os.path.realpath(__file__)) + '/Model_BERT'
#     #
#     #
#     # predictor = ktrain.load_predictor(model_path)
#     # model = ktrain.get_predictor(predictor.model, predictor.preproc)
#     # predicted_sentiment = model.predict(list(df.Tweet.values))
#     # df['predicted_sentiment']=predicted_sentiment
#
#
#     ##########
#     from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification
#     model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
#     model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     classifier = pipeline(task='sentiment-analysis', model=model, tokenizer=tokenizer)
#
#     stars = classifier.predict(list(df.Tweet.values))
#     predicted_sentiment=[x.get('label') for x in stars]
#     df['predicted_sentiment'] = predicted_sentiment
#
#     return df

def visualize(df1):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df=df1
    sns.countplot(data=df,x='predicted_sentiment',y='')





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    translated_df, df_en = translate()
    bert_predict(translated_df,df_en)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
