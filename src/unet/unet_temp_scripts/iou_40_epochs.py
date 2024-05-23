import matplotlib.pyplot as plt
import numpy as np

# load nohup.out file
with open('old_nohup_v2.out', 'r') as f:
    lines = f.readlines()

#keep only the lines starting by Batch: ,Training mIoU: , Validation mIoU: 0.7055814014426594
lines = [line for line in lines if 'Batch ' in line or 'Training mIoU' in line or 'Validation mIoU' in line]

#remove item with Batch size
lines = [line for line in lines if 'Batch size' not in line]

# Batch 1/40, bATCH 2/40 ...
# keep only 1, 2 etc
# if batch in line, remove the "batch ""
lines = [line.replace('Batch ', '') for line in lines]
# plit when / and keep only the first element
lines = [line.split('/')[0] for line in lines]

#filter from lines 

index_list = []
training_mIoU_list = []
validation_mIoU_list = []

# Loop through the lines the last 10 items
lines = lines[:-10]
# print items on lines 3 by 3

my_list = [
    '1', 'Training mIoU: 0.2044754951805821\n', 'Validation mIoU: 0.20508918724954128\n', 
    '2', 'Training mIoU: 0.6379705709579698\n', 'Validation mIoU: 0.6243030192523167\n', 
    '3', 'Training mIoU: 0.6827235035490921\n', 'Validation mIoU: 0.6617568124984873\n', 
    '4', 'Training mIoU: 0.6689694745396331\n', 'Validation mIoU: 0.6549123525619507\n', 
    '5', 'Training mIoU: 0.6866263813347225\n', 'Validation mIoU: 0.6611436099327844\n', 
    '6', 'Training mIoU: 0.6841857155049569\n', 'Validation mIoU: 0.6648826668488568\n',
    '7', 'Training mIoU: 0.7019507455550972\n', 'Validation mIoU: 0.679111000535817\n', 
    '8', 'Training mIoU: 0.699855792075825\n', 'Validation mIoU: 0.6796901215767038\n', 
    '9', 'Training mIoU: 0.7014238765672579\n', 'Validation mIoU: 0.6841935127459723\n', 
    '10', 'Training mIoU: 0.7015604711746963\n', 'Validation mIoU: 0.680336171696926\n', 
    '11', 'Training mIoU: 0.7166980895769356\n', 'Validation mIoU: 0.6989260269650097\n',
    '12', 'Training mIoU: 0.7079861085078215\n', 'Validation mIoU: 0.6858894868657507\n', 
    '13', 'Training mIoU: 0.7153858244419098\n', 'Validation mIoU: 0.6949333438071711\n', 
    '14', 'Training mIoU: 0.7323798419590986\n', 'Validation mIoU: 0.7109986094051394\n', 
    '15', 'Training mIoU: 0.687033265399658\n', 'Validation mIoU: 0.657535717918955\n', 
    '16', 'Training mIoU: 0.7264294371824787\n', 'Validation mIoU: 0.6950829727896328\n', 
    '17', 'Training mIoU: 0.7049164377818533\n', 'Validation mIoU: 0.6855701768192751\n', 
    '18', 'Training mIoU: 0.7418717853617599\n', 'Validation mIoU: 0.7089919334855573\n', 
    '19', 'Training mIoU: 0.6981708568867071\n', 'Validation mIoU: 0.6703816634313814\n', 
    '20', 'Training mIoU: 0.7209753398421175\n', 'Validation mIoU: 0.7010172865000265\n', 
    '21', 'Training mIoU: 0.7224093529607789\n', 'Validation mIoU: 0.6942957346809322\n',
    '22', 'Training mIoU: 0.7357654572392059\n', 'Validation mIoU: 0.7018682609858184\n', 
    '23', 'Training mIoU: 0.733126261907627\n', 'Validation mIoU: 0.711102799351873\n',
    '24', 'Training mIoU: 0.7396906113590218\n', 'Validation mIoU: 0.7059502599054369\n', 
    '25', 'Training mIoU: 0.7503377207761539\n', 'Validation mIoU: 0.7216940825355465\n', 
    '26', 'Training mIoU: 0.7319603582108742\n', 'Validation mIoU: 0.6988002935360218\n', 
    '27', 'Training mIoU: 0.7437602356290954\n', 'Validation mIoU: 0.7145214741086138\n', 
    '28', 'Training mIoU: 0.7498378052835162\n', 'Validation mIoU: 0.7208630861393337\n', 
    '29', 'Training mIoU: 0.7575451239557018\n', 'Validation mIoU: 0.7214162740214117\n', 
    '30', 'Training mIoU: 0.763368139349418\n', 'Validation mIoU: 0.7259061287703186\n', 
    '31', 'Training mIoU: 0.7448090864533307\n', 'Validation mIoU: 0.7120859181058818\n', 
    '32', 'Training mIoU: 0.7632416803658181\n', 'Validation mIoU: 0.7222373120743653\n', 
    '33', 'Training mIoU: 0.7649628506270197\n', 'Validation mIoU: 0.7328248147306771\n', 
    '34', 'Training mIoU: 0.7608568950070428\n', 'Validation mIoU: 0.7228902136457378\n', 
    '35', 'Training mIoU: 0.7650999900930553\n', 'Validation mIoU: 0.7232502033484394\n', 
    '36', 'Training mIoU: 0.7426146953353278\n', 'Validation mIoU: 0.7117445201709353\n', 
    '37', 'Training mIoU: 0.7705085316205917\n', 'Validation mIoU: 0.7304275875975346\n', 
    '38', 'Training mIoU: 0.7589668109368866\n', 'Validation mIoU: 0.7222691016464398\n',
    '39', 'Training mIoU: 0.7751906338961049\n', 'Validation mIoU: 0.7337849525542095\n', 
    '40', 'Training mIoU: 0.7437066603290932\n', 'Validation mIoU: 0.7055814014426594\n']

#plot the training and validation mIoU from ly_list
for i in range(0, len(my_list), 3):
    index_list.append(int(my_list[i]))
    training_mIoU_list.append(round(float(my_list[i+1].split(' ')[-1]), 4))
    validation_mIoU_list.append(round(float(my_list[i+2].split(' ')[-1]),4))

print(len(index_list))
print(len(training_mIoU_list))
print(len(validation_mIoU_list))

# Plot the training and validation mIoU values
plt.plot(index_list, training_mIoU_list, label='Training mIoU')
plt.plot(index_list, validation_mIoU_list, label='Validation mIoU')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.title('Training and Validation mIoU')
plt.legend()
plt.savefig('../unet256_randomshuffling/figures/mious.png')
