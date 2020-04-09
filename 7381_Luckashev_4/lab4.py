from matplotlib import pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import *
from PIL import Image


def build_model(optimizer):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model


def user_image_predict(image_path, model):
    image = np.expand_dims(np.array(Image.open(image_path).convert('L').resize((28, 28))), axis=0)
    image = image / 255
    return model.predict_classes(image)[0]


def line_plot_against(data, name, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    for d in data:
        plt.plot(d)
    ax.set_xticks(np.arange(9))
    ax.set_xticklabels(['100', '50', '20', '10', '1', '0.1', '0.01', '0.001', '0.0001'])
    plt.legend(['RMSprop', 'Adam', 'Nadam', 'Adamax', 'Adagrad', 'Adadelta', 'SGD'])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.tight_layout()
    plt.savefig(name)
    plt.show()
    plt.clf()


def variable_parameters_test(optimizer_list, names, file_prefix, legend):
    i = 0
    for data in optimizer_list:
        name = ''
        for optimizer in data:
            model = build_model(optimizer)
            H = model.fit(train_images, train_labels,
                          epochs=5, batch_size=128,
                          validation_data=(test_images, test_labels),
                          verbose=0
                          )
            plt.subplot(221)
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            plt.plot(H.history['accuracy'])
            plt.subplot(222)
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.plot(H.history['loss'])
            plt.subplot(223)
            plt.xlabel('epochs')
            plt.ylabel('validation accuracy')
            plt.plot(H.history['val_accuracy'])
            plt.subplot(224)
            plt.xlabel('epochs')
            plt.ylabel('validation loss')
            plt.plot(H.history['val_loss'])
        name = names[i]
        i += 1
        plt.legend(legend)
        plt.title(name)
        plt.subplot(221)
        plt.legend(legend)
        plt.title(name)
        plt.subplot(222)
        plt.legend(legend)
        plt.title(name)
        plt.subplot(223)
        plt.legend(legend)
        plt.title(name)
        plt.savefig(file_prefix + name + '.png')
        plt.show()
        plt.clf()


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = (train_images / 255.0, test_images / 255.0)
train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)

# learning rate comparison
train_acc, test_acc = [[],[],[],[],[],[],[]], [[],[],[],[],[],[],[]]
train_loss, test_loss = [[],[],[],[],[],[],[]], [[],[],[],[],[],[],[]]
i = 0
for optimizer in [RMSprop, Adam, Nadam, Adamax, Adagrad, Adadelta, SGD]:
    for lr in [100, 50, 20, 10, 1, 0.1, 0.01, 0.001, 0.0001]:
        model = build_model(optimizer(lr=lr))
        H = model.fit(train_images, train_labels,
                      epochs=5, batch_size=128,
                      validation_data=(test_images, test_labels),
                      verbose=0
                      )
        train_acc[i].append(H.history['accuracy'][-1])
        test_acc[i].append(H.history['val_accuracy'][-1])
        train_loss[i].append(H.history['loss'][-1])
        test_loss[i].append(H.history['val_loss'][-1])
    i += 1

line_plot_against(train_acc, 'training_acc.png', 'Final training accuracy on optimizer', 'Learning rate', 'Accuracy')
line_plot_against(test_acc, 'test_acc.png', 'Validation accuracy on optimizer', 'Learning rate', 'Accuracy')

line_plot_against(train_loss, 'training_loss.png', 'Final training loss on optimizer', 'Learning rate', 'Loss')
line_plot_against(test_loss, 'test_loss.png', 'Validation loss on optimizer', 'Learning rate', 'Loss')

# other parameters
i = 0
optimizer_list = [[], [], []]
for optimizer in [Adam, Adamax, Nadam]:
    for beta in [0.1, 0.5, 0.9]:
        optimizer_list[i].append(optimizer(beta_1=beta))
    i += 1

variable_parameters_test(optimizer_list, ['Adam', 'Adamax', 'Nadam'], 'beta_1_',
                         ['beta_1 = 0.1', 'beta_1 = 0.5', 'beta_1 = 0.9'])

i = 0
optimizer_list = [[], [], []]
for optimizer in [Adam, Adamax, Nadam]:
    for beta in [0.1, 0.5, 0.9]:
        optimizer_list[i].append(optimizer(beta_2=beta))
    i += 1

variable_parameters_test(optimizer_list, ['Adam', 'Adamax', 'Nadam'], 'beta_2_',
                         ['beta_2 = 0.1', 'beta_2 = 0.5', 'beta_2 = 0.9'])

optimizer_list = [[Adam(amsgrad=False), Adam(amsgrad=True)]]
variable_parameters_test(optimizer_list, ['Adam'], 'AMSGrad_', ['default', 'AMSGrad'])

optimizer_list = [[], []]
for momentum in [0.1, 0.5, 0.9, 1.5, 5]:
    optimizer_list[0].append(SGD(nesterov=False, momentum=momentum))
    optimizer_list[1].append(SGD(nesterov=True, momentum=momentum))

variable_parameters_test(optimizer_list, ['SGD', 'SGD(Nesterov)'], 'Momentum_',
                         ['mmntm=0.1', 'mmntm=0.5', 'mmntm=0.9', 'mmntm=1.5', 'mmntm=5'])

optimizer_list = [[], []]
for rho in [0.1, 0.5, 0.9, 1.5, 5]:
    optimizer_list[0].append(RMSprop(rho=rho))
    optimizer_list[1].append(Adadelta(rho=rho))

variable_parameters_test(optimizer_list, ['RMSprop', 'Adadelta'], 'Rho_',
                         ['rho=0.1', 'rho=0.5', 'rho=0.9', 'rho=1.5', 'rho=5.0'])

# user data demonstration
model = build_model(Adam())
H = model.fit(train_images, train_labels,
              epochs=5, batch_size=128,
              validation_data=(test_images, test_labels),
              verbose=0
              )

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc, 'test_loss:', test_loss)

print("actual - 9, predicted -", user_image_predict('9.png', model))
print("actual - 5, predicted -", user_image_predict('5.png', model))
print("actual - 3, predicted -", user_image_predict('3.png', model))
