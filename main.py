import streamlit as st
import plotly.express as px
import pandas as pd
import logging
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import data


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
    st.sidebar.markdown('## ページを選択')
    page = st.sidebar.radio('', ('データ加工', 'データ可視化', 'テストデータ', '決定木'))
    if page == 'データ加工':
        st.session_state.page = 'deal_data'
        logging.info(',%s,ページ選択,%s', st.session_state.username, page)
    elif page == 'データ可視化':
        st.session_state.page = 'vis'
        logging.info(',%s,ページ選択,%s', st.session_state.username, page)
    elif page == 'テストデータ':
        st.session_state.page = 'test'
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
    elif st.session_state.page == 'test':
        test()  
    elif st.session_state.page == 'decision_tree':
        decision_tree()        

# ---------------- usernameの登録 ----------------------------------
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

# ---------------- 訓練データの加工 ----------------------------------
def deal_data():
    st.title("deal_data")

# ---------------- テストデータ　プロット ----------------------------------
def test():
    st.title('テストデータ')

    feature_data = load_num_data()
    full_data = load_full_data()
    label = feature_data.columns



# ---------------- 決定木 : dtreeviz ----------------------------------
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

    tree = data.my_dtree(feature1, feature2)
    st.image(tree, caption=feature1+'_'+feature2)

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
    # st.set_option('deprecation.showPyplotGlobalUse', False)

    # viz = dtreeviz(
    #             clf,
    #             train_X, 
    #             train_y,
    #             target_name='Survived',
    #             feature_names=train_X.columns,
    #             class_names=['Alive', 'Dead'],
    #         ) 
    # st.write("viz OK")

    # viz.view()
    # st.image(viz._repr_svg_(), use_column_width=True)
    # def st_dtree(viz, height=None):
    #     dtree_html = f"<body>{viz.svg()}</body>"
    #     components.html(dtree_html, height=height)

    # st_dtree(viz, 800)
    # st.write('end of code')
    # st.image(viz._repr_svg_(), use_column_width=True)

# ---------------- 可視化 :  各グラフを選択する ----------------------------------
def vis():
    st.title("タイタニック データ")

    feature_data = load_num_data()
    full_data = load_full_data()
    label = feature_data.columns

    st.sidebar.markdown('## 様々なグラフを試してみよう')

    # sidebar でグラフを選択
    graph = st.sidebar.radio(
        'グラフの種類',
        ('棒グラフ', '棒グラフ(色分けあり)', '箱ひげ図', '散布図')
    )

    # 棒グラフ
    # if graph == '棒グラフ':
    #     bar_val = st.selectbox('変数を選択',label)
    #     st.write('生存率と他の変数の関係を調べてみましょう')
    #     fig = px.bar(full_data, x=bar_val, y='Survived')
    #     st.plotly_chart(fig, use_container_width=True)


    # 棒グラフ
    if graph == "棒グラフ":
        st.markdown('## 生存率と他の変数の関係を調べてみる')
        with st.form("棒グラフ"):
            # 変数選択
            hist_val = st.selectbox('変数を選択',label)

            # Submitボタン
            plot_button = st.form_submit_button('グラフ表示')
            if plot_button:
                g = sns.factorplot(data = full_data, x = hist_val, y = 'Survived', kind = 'bar',  ci=None)
                st.pyplot(g)
        # コードの表示
        code = st.sidebar.checkbox('コードを表示')
        if code:
            code_txt = "sns.factorplot(data=full_data, x='" + hist_val + "', y='Survived', kind='bar',  ci=None)"
            st.sidebar.write(code_txt)

    # 棒グラフ: Hue あり
    elif graph == "棒グラフ(色分けあり)":
        label = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        st.markdown('## 生存率と他の変数の関係を調べてみる')
        st.write('性別ごとの分類あり')
        with st.form("棒グラフ(色分けあり)"):
            # 変数選択
            hist_val = st.selectbox('変数を選択',label)
            df = full_data

            # Submitボタン
            plot_button = st.form_submit_button('グラフ表示')
            if plot_button:
                g = sns.factorplot(data = full_data, x = hist_val, y = 'Survived', hue = 'Sex', kind = 'bar',  ci=None)
                st.pyplot(g)
        # コードの表示
        code = st.sidebar.checkbox('コードを表示')
        if code:
            code_txt = "sns.factorplot(data=full_data, x='" + hist_val + "', y='Survived', hue='Sex', kind='bar',  ci=None)"
            st.sidebar.write(code_txt)
    
    # 箱ひげ図
    elif graph == '箱ひげ図':
        st.markdown('## 各変数の分布を箱ひげ図を用いて調べる')
        with st.form("箱ひげ図"):
            # 変数選択
            box_val_y = st.selectbox('箱ひげ図にする変数を選択',label)

            # Submitボタン
            plot_button = st.form_submit_button('グラフ表示')
            if plot_button:
                # 箱ひげ図の表示
                fig = px.box(full_data, x='Survived', y=box_val_y )
                st.plotly_chart(fig, use_container_width=True)
                # コードの表示
        code = st.sidebar.checkbox('コードを表示')
        if code:
            code_txt = "fig = px.box(full_data, x='Survived', y='" + box_val_y + "' )"
            st.sidebar.write(code_txt)
    
    # 散布図
    elif graph == '散布図':
        st.markdown('## 各変数の分布を散布図を用いて調べる')
        with st.form("散布図"):
            left, right = st.beta_columns(2)

            with left: # 変数選択 
                x_label = st.selectbox('横軸を選択',label)

            with right:
                y_label = st.selectbox('縦軸を選択',label)
        
            # Submitボタン
            plot_button = st.form_submit_button('グラフ表示')
            if plot_button:
                # 散布図表示
                fig = px.scatter(full_data,x=x_label,y=y_label)
                st.plotly_chart(fig, use_container_width=True)
        # コードの表示
        code = st.sidebar.checkbox('コードを表示')
        if code:
            code_txt = "fig = px.scatter(full_data,x='" + x_label + "',y='" + y_label + "')"
            st.sidebar.write(code_txt)
 
main()