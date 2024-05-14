import matplotlib.pyplot as plt

epochs = range(1, 18)
# Lists of extracted values
training_losses = [0.7275920256070514, 0.3882693569323515, 0.33705645043843074, 0.33894126383474993, 0.3233038782643997, 0.3174605059091227, 0.31878236419271666, 0.3125920003382033, 0.3143365233492439, 0.3054813909255806, 0.30687947329724213, 0.3046540308754451, 0.29258275310098264, 0.26948120138508613, 0.3343648095252878, 0.2738075146628732, 0.32153426935246765]
validation_losses = [0.7246272584487652, 0.39557117377889567, 0.348489698921812, 0.35626186571758367, 0.3451143388861212, 0.32646778746154803, 0.34127109568437625, 0.3313347064600936, 0.32265771347387084, 0.31763755202550314, 0.3214748722725901, 0.3290897663040408, 0.308685561320905, 0.2927538201790945, 0.3541492258423361, 0.30134304137579326, 0.3464885987341404]
training_mIoU = [0.20466070031741854, 0.6388916517540769, 0.682641305222635, 0.6697143692448091, 0.6877779963208894, 0.6840275767385444, 0.701689152449627, 0.6991494657842158, 0.7012101897073411, 0.7001189777590012, 0.7165033283082484, 0.7081932887563788, 0.716938213158066, 0.7331581400863032, 0.6870823810011234, 0.7261251023935653, 0.7047990118220492]
validation_mIoU = [0.20508866352510863, 0.624288774512965, 0.661749959248921, 0.6549021426973671, 0.6611153658094078, 0.6648959866885481, 0.6791192832691916, 0.6796941926767086, 0.6841750969660694, 0.680333791107967, 0.6989201553936663, 0.6858754959599725, 0.6949820079166313, 0.7110024446557308, 0.6575093004724075, 0.6950740500770766, 0.6855537870834614]

# Plotting training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(epochs, training_losses, label='Training Loss')
plt.plot(epochs, validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

# Plotting training and validation mIoU
plt.figure(figsize=(10, 5))
plt.plot(training_mIoU, label='Training mIoU')
plt.plot(validation_mIoU, label='Validation mIoU')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.title('Training and Validation mIoU')
plt.legend()
plt.show()


'''import matplotlib.pyplot as plt

# Define the epoch range
epochs = range(1, 23)

# Define the training and validation losses
train_losses = [0.5423, 0.4255, 0.3854, 0.3640, 0.3539, 0.3442, 0.3322, 0.3325, 0.3264, 0.3201, 
                0.3196, 0.3170, 0.3080, 0.3107, 0.3089, 0.3095, 0.3003, 0.2973, 0.2954, 0.2940, 
                0.2888, 0.2903]
val_losses = [0.7246, 0.3956, 0.3485, 0.3563, 0.3451, 0.3265, 0.3413, 0.3203, 0.3227, 0.3176, 
              0.3215, 0.3291, 0.3087, 0.2928, 0.3541, 0.3013, 0.3465, 0.3041, 0.3334, 0.3190, 
              0.3135, 0.3005]

# Plot the training and validation losses
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()

plt.show()

import matplotlib.pyplot as plt

# Define the epoch range
epochs = range(1, 23)

# Define the training and validation mIoU values
train_miou = [0.4919, 0.5801, 0.6058, 0.6263, 0.6317, 0.6366, 0.6475, 0.6524, 0.6608, 0.6630,
              0.6665, 0.6707, 0.6812, 0.6751, 0.6848, 0.6802, 0.6893, 0.6898, 0.6878, 0.6912,
              0.7005, 0.6937]
val_miou = [0.2051, 0.6243, 0.6617, 0.6549, 0.6611, 0.6649, 0.6791, 0.6797, 0.6842, 0.6803,
            0.6989, 0.6859, 0.6950, 0.7110, 0.6575, 0.6951, 0.6856, 0.7090, 0.6704, 0.7010,
            0.6943, 0.7019]

# Plot the training and validation mIoU values
plt.plot(epochs, train_miou, label='Training mIoU')
plt.plot(epochs, val_miou, label='Validation mIoU')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.title('Training and Validation mIoU')
plt.legend()

# Show plot
plt.show()

'''