import h5py
import numpy as np
import streamlit as st
import plotly.express as px

classes = [
    "Bean",
    "Bitter Gourd",
    "Bottle Gourd",
    "Brinjal",
    "Broccoli",
    "Cabbage",
    "Capsicum",
    "Carrot",
    "Cauliflower",
    "Cucumber",
    "Papaya",
    "Potato",
    "Pumpkin",
    "Radish",
    "Tomato",
]


def plot(model_name, model_results, metric):
    fig = px.line(
        x=range(1, len(model_results) + 1),
        y=model_results,
        title=f"{model_name} Model {metric}",
    )
    fig.update_yaxes(title_text=metric)
    fig.update_xaxes(title_text="Epochs")
    st.plotly_chart(fig, use_container_width=True)


st.set_page_config(page_title="Graphs", page_icon="📈", layout="centered")
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
    \nSome CNN implementations on Kaggle have 91-95% test accuracy. But we got only 71% test accuracy. The reason is that we didn't made any hyperparameter optimization. If we do that, we can get better results.
	"""
)

# plot cnn model loss
plot("CNN", test_loss_cnn, "Loss")

# plot cnn model accuracy
plot("CNN", test_acc_cnn, "Accuracy")

conf_matrix = np.load("files/cnn_conf_matrix.npy")
fig = px.imshow(
    conf_matrix,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=classes,
    y=classes,
    title="CNN Confusion Matrix",
)
st.plotly_chart(fig, use_container_width=True)

st.write("---")

st.subheader("ResNet Model Results")
st.write(
    """
    We used ResNet18 as a pretrained model. We didn't train it. We just changed the last layer of the model and finetuned it with our data.
	\nThe ResNet18 model was trained with a batch size of 64, learning rate of 0.001 and 100 epochs. But the model was converge faster. So, the program stopped the training at 22nd epoch.
    \nWe got 99% test accuracy with ResNet18 model. It's a great result. But even we applied data augmentation to train data and test results are better than train, I think the model is overfiting. Because images in the train and test data are very similar.
	"""
)

# plot resnet model loss
plot("ResNet", test_loss_resnet, "Loss")

# plot resnet model accuracy
plot("ResNet", test_acc_resnet, "Accuracy")

conf_matrix = np.load("files/resnet_conf_matrix.npy")
fig = px.imshow(
    conf_matrix,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=classes,
    y=classes,
    title="ResNet Confusion Matrix",
)
st.plotly_chart(fig, use_container_width=True)
