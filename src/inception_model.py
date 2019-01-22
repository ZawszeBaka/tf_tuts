import helper_functions
from helper_functions import *

from pprint import pprint

# functions and classes for loading and using Inception model
import inception
from IPython.display import Image, display

# Download the Inception Model and store to datasets/inception
# inception.maybe_download()
print('[INFO] Inception path', inception.data_dir)

# Load the Inception Model
model = inception.Inception()

def classify(image_path):
    print('\n[INFO] Image path', image_path)

    # Use the Inception model to classify the image
    pred = model.classify(image_path=image_path)

    # Print the scores and names for the top-10 prediction
    model.print_scores(pred=pred, k=10, only_first_name=True)

    # Display the image
    img = cv2.imread(image_path)
    cv2.imshow('Image', img)
    cv2.waitKey()

if __name__ == '__main__':
    classify('../images/hulk.jpg')
    classify('../images/parrot.jpg')
    classify('../images/parrot_cropped1.jpg')
    classify('../images/style1.jpg')
    classify('../images/willy_wonka_new.jpg')

    cv2.destroyAllWindows()
