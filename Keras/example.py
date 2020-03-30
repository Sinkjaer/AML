import numpy as np
import matplotlib.pyplot as plt
import os
import sys

os.chdir(sys.path[0]) # set the working directory 
# from lstm_vae import create_lstm_vae
from vae import create_lstm_vae

def get_data():
    # read data from file
    data = np.fromfile('sample_data.dat').reshape(419,13)
    timesteps = 3
    dataX = []
    for i in range(len(data) - timesteps - 1):
        x = data[i:(i+timesteps), :]
        dataX.append(x)
    return np.array(dataX)


if __name__ == "__main__":
    x = get_data()
    input_dim = x.shape[-1] # 13
    timesteps = x.shape[1] # 3
    batch_size = 1

    vae, enc, gen = create_lstm_vae(input_dim, 
        timesteps=timesteps, 
        batch_size=batch_size, 
        intermediate_dim=2,
        latent_dim=2,
        epsilon_std=1.)

    vae.fit(x, x, epochs=10)

    preds = vae.predict(x, batch_size=batch_size)

    # pick a column to plot.
    print("[plotting...]")
    print("x: %s, preds: %s" % (x.shape, preds.shape))
    plt.plot(x[:,0,3], label='data')
    plt.plot(preds[:,0,3], label='predict')
    plt.legend()
    plt.show()


