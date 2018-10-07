import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from pylab import *
from data import *

from donkeycar.parts.datastore import TubGroup, TubHandler

def show(fig, subplace, title, _img):
    ax = fig.add_subplot(*subplace)
    ax.axis('off')
    ax.set_title(title)
    ax.imshow(_img)

def save_3_views(images, steering_angle, name, brightness=0, shadow=0):
    fig = plt.figure(figsize=(12, 20))
    titles = ["Center image. Steering angle = " + str(steering_angle) ,
              "Left image. Steering angle = " + str(steering_angle + parameter['steering_bias']),
              "Right image. Steering angle = " + str(steering_angle - parameter['steering_bias'])]
    for i in range(0, images.shape[0]):
        image = load_image(images[i])
        if (brightness):
            image = random_brightness(image)
        if (shadow):
            image = random_shadow(image)
        show(fig, (3, 1, i+1), titles[i], image)
    plt.tight_layout()
    savefig(parameter['saved_images_folder'] + name)

def save_flip_view(images, name):
    fig = plt.figure(figsize=(12, 8))
    image = load_image(images[0])
    show(fig, (2,1,1), "Original", image)
    image, _ = horizontal_flip(image, 0)
    show(fig, (2,1,2), "Flipped", image)
    plt.tight_layout()
    savefig(parameter['saved_images_folder'] + name)

def steering_angles_histogram(steering_angles, name, title, bins='auto', raw=0, fully_augmented=0):
    fig = plt.figure(figsize=(12, 8))
    if not raw:
        steering_angles = np.append(steering_angles, steering_angles * -1.)
        if fully_augmented:
            steering_angles = np.append(steering_angles, steering_angles + parameter['steering_bias'])
            steering_angles = np.append(steering_angles, steering_angles - parameter['steering_bias'])
    plt.hist(steering_angles, bins=bins)  # arguments are passed to np.histogram
    plt.title(title)
    savefig(parameter['saved_images_folder'] + name)

def test_image_shift(images, steering_angle, name, j=0):
    fig = plt.figure(figsize=(6, 8))
    for i in range(images.shape[0]):
        image = load_image(images[i][j])   # Load image
        if j==0:
            angle = steering_angle[i]
        elif j==1:
            angle = steering_angle[i] + parameter['steering_bias']
        elif j==2:
            angle = steering_angle[i] - parameter['steering_bias']
        show(fig, (images.shape[0],2,i*2+1), "Original - steering_angle = {0:.3f}".format(angle), image)
        image, angle = random_shift(image, angle)
        show(fig, (images.shape[0],2,i*2+2), "Shifted - steering_angle = {0:.3f}".format(angle), image)
    plt.tight_layout()
    savefig(parameter['saved_images_folder'] + name)

def get_random_image_id(image_paths):
    '''
    Returns a random number within the range of images available
    '''
    return int(np.random.uniform()*image_paths.shape[0])

class test_model(object):
    '''
    The test_model class allows me to visualize the steering_angle
    predicted by the CNN for a given picture.
    This helps me figuring out where the problems might come from.
    '''
    def __init__(self, parameter):
        self.model = load_model(parameter['saved_model'])

    def make_prediction(self, image, name):
        image_ready = preprocess(image)
        steering_angle = float(self.model.predict(image_ready[None, :, :, :], batch_size=1))
        fig = plt.figure(figsize=(6, 4))
        show(fig, (1, 1, 1), "Predicted steering angle : {}".format(steering_angle), image)
        savefig(parameter['saved_images_folder'] + name)

    def load_N_make_prediction(self, path, name):
        image = load_image(path[0])
        self.make_prediction(image, name)


if __name__ == "__main__":

    X_keys = ['cam/image_array']
    y_keys = ['user/angle', 'user/throttle']
    tubgroup = TubGroup("./data/tub_29_18-09-09")
    train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys,
                                                    batch_size=32,
                                                    train_frac=0.8)
    print("Fetching data ...")
    steering_angles = []
    for _, user_data in train_gen:
        steering_angles.append(user_data[0])

    print("%f frames.".format(len(steering_angles)))

    if 0:
        i = get_random_image_id(image_paths)
        save_3_views(image_paths[i], steering_angles[i], '3views.png')

    if 0:
        i = get_random_image_id(image_paths)
        save_3_views(image_paths[i], steering_angles[i], '3bright.png', brightness=1)
        i = get_random_image_id(image_paths)
        save_3_views(image_paths[i], steering_angles[i], '3shadow.png', shadow=1)

        i = get_random_image_id(image_paths)
        save_flip_view(image_paths[i], "flip.png")

    if 1:
        steering_angles_histogram(steering_angles, 'raw_histo.png', "Histogram of the raw data.", raw=1)
        #steering_angles_histogram(steering_angles, 'histo.png', "Histogram of the data when reversed horizontally.")
        #steering_angles_histogram(steering_angles, 'full_histo.png', "Histogram of the data when using side cameras and a bias of " + str(parameter['steering_bias']), fully_augmented=1)

    if 0:
        i = [get_random_image_id(image_paths) for _ in range(4)]
        test_image_shift(image_paths[i], steering_angles[i], 'shift.png')
        test_image_shift(image_paths[i], steering_angles[i], 'shift_left.png', 1)

    if 0:
        steering_avg = 0
        gen = batch_generator(image_paths, steering_angles, parameter, True)
        steers = np.zeros(1)
        nb_batchs = 40
        for i in range(0, nb_batchs):
            _, steers_new = next(gen)
            steers = np.append(steers, steers_new)
            steering_avg += steers_new.sum()
        steering_angles_histogram(steers, 'gen_histo_true_ramdom',
                'Histogram of the generator data : ' + str(steers.shape[0]) + ' samples. Average steer : ' + str(steering_avg / (nb_batchs * parameter['batch_size'])),
                raw=1, bins=60)

    if 0:
        model = test_model(parameter)

        i = get_random_image_id(image_paths)
        model.load_N_make_prediction(image_paths[i], "pred1.png")
        i = get_random_image_id(image_paths)
        model.load_N_make_prediction(image_paths[i], "pred2.png")
        i = get_random_image_id(image_paths)
        model.load_N_make_prediction(image_paths[i], "pred3.png")
        i = get_random_image_id(image_paths)
        model.load_N_make_prediction(image_paths[i], "pred4.png")
        i = get_random_image_id(image_paths)
        model.load_N_make_prediction(image_paths[i], "pred5.png")

    print("End of test.")
