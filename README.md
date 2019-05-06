# Skin Cancer Image Classifier
My first attempt at a machine learning API, using a pre-calculated model trained using [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) data

server.py is a very tiny [Starlette API](https://www.starlette.io/) server which simply accepts file image uploads and runs them against the pre-calculated model.

Go to this url and classify the images.
https://skincancer-59r3.onrender.com/
