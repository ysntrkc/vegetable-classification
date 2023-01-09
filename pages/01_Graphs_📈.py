import h5py
import streamlit as st
import plotly.express as px


def plot(model_name, model_results, metric):
    fig = px.line(
        x=range(1, len(model_results) + 1),
        y=model_results,
        title=f"{model_name} Model {metric}",
    )
    fig.update_yaxes(title_text=metric)
    fig.update_xaxes(title_text="Epochs")
    st.plotly_chart(fig, use_container_width=False)


st.set_page_config(page_title="Graphs", page_icon="📈", layout="wide")
st.title("Model Results")

st.write("---")

cnn_model_results = h5py.File("results/cnn.h5", "r")
resnet_model_results = h5py.File("results/resnet.h5", "r")

test_loss_cnn = cnn_model_results["test_loss"][cnn_model_results["test_loss"][:] != 0]
test_acc_cnn = cnn_model_results["test_acc"][cnn_model_results["test_acc"][:] != 0]

test_loss_resnet = resnet_model_results["test_loss"][
    resnet_model_results["test_loss"][:] != 0
]
test_acc_resnet = resnet_model_results["test_acc"][
    resnet_model_results["test_acc"][:] != 0
]

st.subheader("CNN Model Results")
st.write(
    """
	The CNN model was trained with a batch size of 64, learning rate of 0.001 and 100 epochs. But the model was converge faster. So, the program stopped the training at 30th epoch.
	"""
)

col_1, _, col_2, _ = st.columns([6, 1, 6, 2])

with col_1:
    # plot cnn model loss
    plot("CNN", test_loss_cnn, "Loss")

with col_2:
    # plot cnn model accuracy
    plot("CNN", test_acc_cnn, "Accuracy")

st.write("---")

st.subheader("ResNet Model Results")
st.write(
    """
	The ResNet18 model was trained with a batch size of 64, learning rate of 0.001 and 100 epochs. But the model was converge faster. So, the program stopped the training at 22nd epoch.
	"""
)

col_1, _, col_2, _ = st.columns([6, 1, 6, 2])

with col_1:
    # plot resnet model loss
    plot("ResNet", test_loss_resnet, "Loss")

with col_2:
    # plot resnet model accuracy
    plot("ResNet", test_acc_resnet, "Accuracy")
