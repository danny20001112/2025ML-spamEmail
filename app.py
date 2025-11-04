import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 以下套件為可選，若未安裝則降級顯示
try:
    import seaborn as sns
except Exception:
    sns = None
try:
    import plotly.express as px
except Exception:
    px = None
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None
try:
    import jieba
except Exception:
    jieba = None

@st.cache_data
def load_data():
    # 優先載入 spam.csv，再試 test_spam.csv，最後內建示例
    paths = ['spam.csv', 'test_spam.csv']
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, encoding='latin-1', low_memory=False)
                if {'v1','v2'}.issubset(df.columns):
                    df = df[['v1','v2']].dropna(subset=['v1','v2'])
                    df.columns = ['label','text']
                    df['text'] = df['text'].astype(str)
                    return df
            except Exception:
                continue
    # 內建示例（最小可用集）
    sample = pd.DataFrame({
        'label': ['ham','spam','ham','spam','ham','spam'],
        'text': [
            '明天下午 2 點開會，請準時。',
            '恭喜你！中獎了，點擊領取獎品。',
            '請幫我確認附件文件內容。',
            '限時優惠 50% 折扣，立即搶購！',
            '午餐要訂什麼？',
            '您被選中獲得免費旅遊，點此申請。'
        ]
    })
    return sample

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    if sns:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['ham','spam'], yticklabels=['ham','spam'])
    else:
        ax.imshow(cm, cmap='Blues')
        for (i,j),v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha='center', va='center', color='white')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['ham','spam']); ax.set_yticklabels(['ham','spam'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('Confusion Matrix')
    return fig

def make_wordcloud(text):
    if WordCloud is None:
        return None
    text_joined = ' '.join(text)
    font_path = None
    if os.name == 'nt':
        candidate = r"C:\Windows\Fonts\msyh.ttc"
        if os.path.exists(candidate):
            font_path = candidate
    try:
        if font_path:
            wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(text_joined)
        else:
            wc = WordCloud(width=800, height=400, background_color='white').generate(text_joined)
        return wc
    except Exception:
        return None

def safe_report_dict(y_test, y_pred):
    try:
        return classification_report(y_test, y_pred, output_dict=True)
    except Exception:
        return {}

def main():
    st.set_page_config(page_title="垃圾郵件分類器", layout="wide")
    st.title("📧 智慧垃圾郵件分類器")
    st.write("系統會自動載入 spam.csv 或 test_spam.csv，若無則使用內建示例。")

    data = load_data()
    if data is None or data.empty:
        st.error("無可用資料")
        st.stop()

    # 側邊欄：模型設定與資料統計
    with st.sidebar:
        st.header("⚙️ 設定")
        min_df = st.slider("Tfidf min_df", 1, 5, 1)
        test_size = st.slider("測試集比例", 0.1, 0.4, 0.2)
        st.markdown("---")
        st.subheader("資料統計")
        st.write(f"資料筆數：{len(data)}")
        spam_count = int((data['label']=='spam').sum())
        st.write(f"垃圾郵件數：{spam_count}  ({spam_count/len(data):.1%})")
        st.markdown("---")
        st.write("關於：版本 1.0")

    # 前處理與訓練
    X_text = data['text'].astype(str)
    y = (data['label']=='spam').astype(int)
    vectorizer = TfidfVectorizer(min_df=min_df)
    try:
        X = vectorizer.fit_transform(X_text)
    except Exception as e:
        st.error(f"向量化失敗：{e}")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # 評估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = safe_report_dict(y_test, y_pred)

    st.markdown("## 模型摘要")
    c1, c2, c3 = st.columns(3)
    c1.metric("準確率", f"{acc:.2%}")
    prec = report.get('1', {}).get('precision') if isinstance(report, dict) else report.get('spam',{}).get('precision',0.0)
    rec = report.get('1', {}).get('recall') if isinstance(report, dict) else report.get('spam',{}).get('recall',0.0)
    c2.metric("垃圾郵件精確度", f"{(prec or 0):.2%}")
    c3.metric("垃圾郵件召回率", f"{(rec or 0):.2%}")

    # 主區：輸入與預測
    st.markdown("---")
    st.subheader("📝 郵件分析")
    user_text = st.text_area("貼上郵件內容，按「分析」：", height=180)
    if st.button("分析"):
        if not user_text.strip():
            st.warning("請輸入內容")
        else:
            with st.spinner("分析中..."):
                time.sleep(0.5)
                vec = vectorizer.transform([user_text])
                pred = model.predict(vec)[0]
                proba = model.predict_proba(vec)[0]
                if pred == 1:
                    st.error("⚠️ 可能是垃圾郵件")
                else:
                    st.success("✅ 可能是正常郵件")
                # 顯示機率
                labels = ['正常','垃圾']
                probs = [proba[0], proba[1]]
                if px:
                    dfp = pd.DataFrame({'類別':labels,'機率':probs})
                    figp = px.bar(dfp, x='類別', y='機率', color='類別',
                                  color_discrete_map={'正常':'green','垃圾':'red'}, range_y=[0,1])
                    st.plotly_chart(figp, use_container_width=True)
                else:
                    fig, ax = plt.subplots()
                    ax.bar(labels, probs, color=['green','red'])
                    ax.set_ylim(0,1)
                    st.pyplot(fig)
                # 進階：文字雲（若可）
                st.markdown("### 🔍 進階分析")
                if WordCloud is not None:
                    tokens = jieba.lcut(user_text) if jieba else user_text.split()
                    wc = make_wordcloud(tokens)
                    if wc:
                        fig_wc = plt.figure(figsize=(8,3))
                        plt.imshow(wc, interpolation='bilinear')
                        plt.axis('off')
                        st.pyplot(fig_wc)
                        plt.close()
                    else:
                        st.info("無法產生文字雲（缺少資源或字型）")
                else:
                    st.info("未安裝 wordcloud，無法顯示文字雲。")

    # 模型評估詳細
    st.markdown("---")
    st.subheader("📊 模型評估")
    cm_fig = plot_confusion(y_test, y_pred)
    st.pyplot(cm_fig)
    st.code(classification_report(y_test, y_pred, target_names=['ham','spam']))

if __name__ == "__main__":
    main()
