import glob
import streamlit as st

st.set_page_config(page_title="Dataset", page_icon="📂", layout="centered")
st.title("Dataset")

st.write("---")

st.write(
    "**Dataset Info:** The dataset used for this project was collected from Kaggle. It contains 15000 images for train, 3000 for validation and 3000 for test. The dataset can be found [here](https://www.kaggle.com/misrakahmed/vegetable-image-dataset)."
)

st.subheader("Sample Images from the 15 Classes of Vegetables:")

sample_images = glob.glob("images/*.jpg")
col1, _, col2, _, col3 = st.columns([4, 1, 4, 1, 4])

for i in range(0, len(sample_images), 3):
    with col1:
        caption = sample_images[i].split("\\")[-1].split(".")[0].split("/")[-1]
        st.image(sample_images[i], width=150, caption=caption)
        st.write("")
    with col2:
        caption = sample_images[i + 1].split("\\")[-1].split(".")[0].split("/")[-1]
        st.image(sample_images[i + 1], width=150, caption=caption)
        st.write("")
    with col3:
        caption = sample_images[i + 2].split("\\")[-1].split(".")[0].split("/")[-1]
        st.image(sample_images[i + 2], width=150, caption=caption)
        st.write("")
