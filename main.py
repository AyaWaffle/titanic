import streamlit as st
import plotly.express as px
import pandas as pd
import logging
from sklearn.metrics import accuracy_score
# from dtreeviz.trees import dtreeviz
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import dtreeviz
import streamlit.components.v1 as components
import graphviz as graphviz

st.set_page_config(
    # page_title="PE Score Analysis App",
    # page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s,%(message)s")


DATA_SOURCE = './data/titanic.csv'

@st.cache
def load_full_data():
    data = pd.read_csv(DATA_SOURCE)
    return data

@st.cache 
def load_num_data():
    data = pd.read_csv(DATA_SOURCE)
    rows = ['Survived']
    data = data.drop(rows, axis=1)
    return data

# @st.cache 
# def load_filtered_data(data, genre_filter):
#     # 数値でフィルター(何点以上)
#     # filtered_data = data[data['num_rooms'].between(rooms_filter[0], rooms_filter[1])]
#     grade_filter = []
#     gender_filter = []
#     for elem in genre_filter:
#         grade_filter.append(str(elem[0:2]))
#         gender_filter.append(str(elem[2]))

#     filtered_data = data[data['学年'].isin(grade_filter)]
#     filtered_data = filtered_data[filtered_data['性別'].isin(gender_filter)]

#     return filtered_data

@st.cache
def load_ML_data(feature1, feature2, train_num = 600):
    df = load_full_data()
    # X = df.drop('Survived', axis=1)  # XはSurvivedの列以外の値
    X = df[[feature1, feature2]]
    y = df.Survived  # yはSurvivedの列の値

    train_num = 600
    train_X = X[:train_num]
    test_X = X[train_num:]
    train_y = y[:train_num]
    test_y = y[train_num:]
    return (train_X, test_X, train_y, test_y)


def main():
    # # If username is already initialized, don't do anything
    # if 'username' not in st.session_state or st.session_state.username == 'default':
    #     st.session_state.username = 'default'
    #     input_name()
    #     st.stop()
    if 'username' not in st.session_state:
        st.session_state.username = 'test'
            
    if 'page' not in st.session_state:
        # st.session_state.page = 'input_name' # usernameつける時こっち
        st.session_state.page = 'deal_data'


    # --- page選択ラジオボタン
    page = st.sidebar.radio('ページ選択', ('データ加工', 'データ可視化', '決定木'))
    if page == 'データ加工':
        st.session_state.page = 'deal_data'
        logging.info(',%s,ページ選択,%s', st.session_state.username, page)
    elif page == 'データ可視化':
        st.session_state.page = 'vis'
        logging.info(',%s,ページ選択,%s', st.session_state.username, page)
    elif page == '決定木':
        st.session_state.page = 'decision_tree'
        logging.info(',%s,ページ選択,%s', st.session_state.username, page)

    # --- page振り分け
    if st.session_state.page == 'input_name':
        input_name()
    elif st.session_state.page == 'deal_data':
        deal_data()
    elif st.session_state.page == 'vis':
        vis()        
    elif st.session_state.page == 'decision_tree':
        decision_tree()        

def input_name():
    # Input username
    with st.form("my_form"):
        inputname = st.text_input('username', 'ユーザ名')
        submitted = st.form_submit_button("Submit")
        if submitted: # Submit buttonn 押された時に
            if inputname == 'ユーザ名' or input_name == '': # nameが不適当なら
                submitted = False  # Submit 取り消し

        if submitted:
            st.session_state.username = inputname
            st.session_state.page = 'deal_data'
            st.write("名前: ", inputname)
    
def deal_data():
    st.title("deal_data")

# ---------------- 可視化 :  各グラフを選択する ----------------------------------
def decision_tree():
    st.title("生存できるか予測しよう")
    
    st.write('予測に使う変数を2つ選ぼう')
    left, right = st.beta_columns(2)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']
    with left:
        feature1 = st.selectbox('予測に使う変数1',features)
    with right:
        feature2 = st.selectbox('予測に使う変数2',features)

    started = st.button('学習スタート')
    if not started: 
        st.stop()
    
    # データの取得
    train_X, test_X, train_y, test_y = load_ML_data(feature1, feature2, train_num = 600)

    # 木の深さを3に制限
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    # 学習
    clf = clf.fit(train_X, train_y)

    # test_Xデータを全部予測する
    pred = clf.predict(test_X)
    # 正解率を計算する
    acc = accuracy_score(pred, test_y)

    st.success('学習終了！！')

    st.write(f'accuracy: {acc:.5f}')

    vis_tree = st.button('決定木をみてみる')

    # if vis_tree:
    #     viz = dtreeviz(
    #             clf,
    #             train_X, 
    #             train_y,
    #             target_name='Survived',
    #             feature_names=train_X.columns,
    #             class_names=['Alive', 'Dead'],
    #         ) 

    #     viz.view()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    viz = dtreeviz(
                clf,
                train_X, 
                train_y,
                target_name='Survived',
                feature_names=train_X.columns,
                class_names=['Alive', 'Dead'],
            ) 
    st.write("viz OK")

    # viz.view()
    # st.image(viz._repr_svg_(), use_column_width=True)
    def st_dtree(viz, height=None):
        dtree_html = f"<body>{viz.svg()}</body>"
        components.html(dtree_html, height=height)

    st_dtree(viz, 800)
    st.write('end of code')






# ---------------- 可視化 :  各グラフを選択する ----------------------------------
def vis():
    st.title("タイタニック データ")

    feature_data = load_num_data()
    full_data = load_full_data()
    label = feature_data.columns

    st.sidebar.markdown('## いろんなグラフを試してみよう')

    # sidebar でグラフを選択
    graph = st.sidebar.radio(
        'グラフの種類',
        ('棒グラフ', 'ヒストグラム', '箱ひげ図')
    )

    # 棒グラフ
    if graph == '棒グラフ':
        bar_val = st.selectbox('変数を選択',label)
        st.write('生存率と他の変数の関係を調べてみましょう')
        fig = px.bar(full_data, x=bar_val, y='Survived')
        st.plotly_chart(fig, use_container_width=True)


    # ヒストグラム
    elif graph == "ヒストグラム":
        hist_val = st.selectbox('変数を選択',label)
        fig = px.histogram(feature_data, x=hist_val)
        st.plotly_chart(fig, use_container_width=True)
    
    # 箱ひげ図
    elif graph == '箱ひげ図':
        box_val_y = st.selectbox('箱ひげ図にする変数を選択',label)

        fig = px.box(full_data, x='Survived', y=box_val_y )
        st.plotly_chart(fig, use_container_width=True)

      

main()