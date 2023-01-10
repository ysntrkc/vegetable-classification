import streamlit as st

st.set_page_config("About Project", "📖", "centered")

st.title("About Project")

st.write("---")

st.subheader("Problem Description")
st.write(
    "Vegetables may need some classification while transferring them or before processing them. We don’t want to classify them using a human worker since we can do it more quickly with a computer. Therefore, in our problem we are going to do image classification on a vegetable dataset that contains 15 different classes."
)

st.subheader("Solution Approach")
st.markdown(
    """
    We are going to classify the images using CNN and RESNET-18 structure. CNN makes image processing easier by providing us with some useful functions to extract information about images.
* We can apply convolution operation to detect patterns in images such as edges, shapes etc.
* We can apply pooling operations to reduce the image size significantly to make calculations faster.
* Lastly, we can pass the parameters through a fully connected layer for classification.
	\nResidual networks are used for complex image classification tasks. Unlike CNN it doesn’t have to pass through all the layers contained in the neural network. It can skip some of the layers and that gives us a faster training process since we don’t have to make all the calculations in all of the layers. In the project we used a pretrained version of RESNET-18. It is trained by a dataset called ImageNet which is a very large image dataset. Compared to our CNN model the accuracy of RESNET-18 is really high.
            """
)

st.subheader("Data Preparation")
st.write(
    """
	The dataset consists of 21000 images and the size of each image is 224x224 in .jpg format. It contains 3 folders for each class as train, test and validation. We split the dataset as %70 for train, %15 for test and %15 for validation. In total we have 15 different classes such as: Bean, bitter gourd, bottle gourd, brinjal, broccoli, cabbage, capsicum, carrot, cauliflower, cucumber, papaya, potato, pumpkin, radish and tomato.
	Since the size is too large, in preprocessing step we resize all of the images to 150x150. We also applied some augmentation methods to improve our model with different versions of the images. We applied rotation, horizontal and vertical flip, color filter, blur."""
)
