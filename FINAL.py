##Step 1: Load packages
import tensorflow as tf
import FINAL_test_w_first_copy
import numpy as npx
import pandas as pd
import numpy as np
import warnings
from collections import OrderedDict
import pylogit as pl

from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm
from numpy.random import seed
from sklearn.model_selection import train_test_split

# Set the number of latent classes and iterations (the number of iterations it is not used in this version of the code)
n_classes = 3
ITER= 20
n_latent_variables = 3

#number of epoch of the NN of the classes
n_epochs= 50
n_nodes= 6

filename = 'Final'+str(n_classes)+'latent'+ str(n_latent_variables)+'4.0_first_copy_new'


#### Take the data
with open('data.pickle', 'rb') as handle:
    data = pickle.load(handle)


with open('data_test.pickle', 'rb') as handle:
    data_test = pickle.load(handle)


## Step 3: Convert the data from wide to long format
# Create the list of individual specific variables (variables that are specific to an individual 
# and do not vary across alternatives)
ind_variables = data.columns.tolist()[:100]
ind_variables.remove('Choice')

# Specify the variables that vary across individuals and some or all alternatives
# The keys are the column names that will be used in the long format dataframe
# The values are dictionaries whose key-value pairs are the alternative id and
# the column name of the corresponding column that encodes that variable for
# the given alternative.
alt_varying_variables = {u'SUB_EURO': dict([(1, 'Alt1_SUB_EURO'),
                                               (2, 'Alt2_SUB_EURO'),
                                               (3, 'Alt3_SUB_EURO'),
                                               (4, 'Alt4_SUB_EURO')]),
                        u'USECOST_EURO': dict([(1, 'Alt1_USECOST_EURO'),
                                               (2, 'Alt2_USECOST_EURO'),
                                               (3, 'Alt3_USECOST_EURO'),
                                               (4, 'Alt4_USECOST_EURO')]),
                        u'HOUR_PACKAGE': dict([(1, 'Alt1_HOUR_PACKAGE'),
                                               (2, 'Alt2_HOUR_PACKAGE'),
                                               (3, 'Alt3_HOUR_PACKAGE'),
                                               (4, 'Alt4_HOUR_PACKAGE')]),
                         u'DAY_PACKAGE': dict([(1, 'Alt1_DAY_PACKAGE'),
                                               (2, 'Alt2_DAY_PACKAGE'),
                                               (3, 'Alt3_DAY_PACKAGE'),
                                               (4, 'Alt4_DAY_PACKAGE')]),
                        u'ENG_COMB': dict([(1, 'Alt1_ENG_COMB'),
                                               (2, 'Alt2_ENG_COMB'),
                                               (3, 'Alt3_ENG_COMB'),
                                               (4, 'Alt4_ENG_COMB')]),
                        u'ENG_ELECTR': dict([(1, 'Alt1_ENG_ELETR'),
                                               (2, 'Alt2_ENG_ELETR'),
                                               (3, 'Alt3_ENG_ELETR'),
                                               (4, 'Alt4_ENG_ELETR')]),
                        u'SMLLSDN': dict([(1, 'Alt1_SMLLSDN'),
                                               (2, 'Alt2_SMLLSDN'),
                                               (3, 'Alt3_SMLLSDN'),
                                               (4, 'Alt4_SMLLSDN')]),
                        u'SMLLSDNSUV': dict([(1, 'Alt1_SMLLSDNSUV'),
                                               (2, 'Alt2_SMLLSDNSUV'),
                                               (3, 'Alt3_SMLLSDNSUV'),
                                               (4, 'Alt4_SMLLSDNSUV')]),
                        u'WLKTIMEACC': dict([(1, 'Alt1_WLKTIMEACC'),
                                               (2, 'Alt2_WLKTIMEACC'),
                                               (3, 'Alt3_WLKTIMEACC'),
                                               (4, 'Alt4_WLKTIMEACC')]),
                        u'BOOKING': dict([(1, 'Alt1_BOOKING'),
                                               (2, 'Alt1_BOOKING'),
                                               (3, 'Alt1_BOOKING'),
                                               (4, 'Alt1_BOOKING')]),
                        u'WLKTIMEPARK': dict([(1, 'Alt1_WLKTIMEPARK'),
                                               (2, 'Alt2_WLKTIMEPARK'),
                                               (3, 'Alt3_WLKTIMEPARK'),
                                               (4, 'Alt4_WLKTIMEPARK')]),
                        u'PROBCAR': dict([(1, 'Alt1_PROBCAR'),
                                               (2, 'Alt2_PROBCAR'),
                                               (3, 'Alt3_PROBCAR'),
                                               (4, 'Alt4_PROBCAR')])}

# Specify the availability variables
# Note that the keys of the dictionary are the alternative id's.
# The values are the columns denoting the availability for the
# given mode in the dataset.
availability_variables = {1: 'Av',
                          2: 'Av',
                          3: 'Av',
                          4: 'Av',
                          5: 'Av'}


# Determine the columns for: alternative ids, the observation ids and the choice

# The 'custom_alt_id' is the name of a column to be created in the long-format data
# It will identify the alternative associated with each row.
custom_alt_id = "mode_id"

# Create a custom id column that ignores the fact that this is a 
# panel/repeated-observations dataset. Note the +1 ensures the id's start at one.
obs_id_column = "custom_id"
data[obs_id_column] = np.arange(data.shape[0],
                                            dtype=int) + 1


# Create a variable recording the choice column
choice_column = "Choice"

# Perform the conversion to long-format
long_data = pl.convert_wide_to_long(data, ind_variables, 
                                           alt_varying_variables, 
                                           availability_variables, 
                                           obs_id_column, 
                                           choice_column,
                                           new_alt_id_name=custom_alt_id)

###Same but for the test

ind_variables = data_test.columns.tolist()[:100]
ind_variables.remove('Choice')

data_test[obs_id_column] = np.arange(data_test.shape[0],
                                            dtype=int) + 1

# Perform the conversion to long-format
long_data_test = pl.convert_wide_to_long(data_test, ind_variables, 
                                           alt_varying_variables, 
                                           availability_variables, 
                                           obs_id_column, 
                                           choice_column,
                                           new_alt_id_name=custom_alt_id)


#select the columns for the sociocharacteristics
Q = long_data[['ID','custom_id','CSMEMBER', 'KIDSUPTO12Y', 'p_Age', 'CARATHOME', 'BIKEATHOME',
                'INCOME_LOW', 'INCOME_MEDIUM', 'INCOME_HIGH',
                'STUDENT', 'EMPLOYED', 'RETIRED', 'WOMAN', 
               'pcaratt_1_resp', 'pcaratt_2_resp', 'pcaratt_3_resp', 'pcaratt_4_resp',
               'pcaratt_5_resp', 'pcaratt_6_resp', 'pcaratt_7_resp', 'cs_feel_1_resp',
               'cs_feel_2_resp', 'cs_feel_3_resp', 'cs_feel_4_resp', 'cs_feel_5_resp',
               'cs_feel_6_resp', 'cs_feel_7_resp', 'cs_feel_8_resp', 'cs_feel_9_resp', 'cs_feel_10_resp']]


#Drop the rows from the same choice
Q.drop_duplicates(subset ="custom_id",keep= 'first', inplace = True)
#Drop the duplicates from the same decision maker
Q.drop_duplicates(subset ="ID",keep= 'first', inplace = True)
Q= Q.drop(columns=['custom_id'])

#Scale age
Q['p_Age'] = preprocessing.scale(Q['p_Age'])

#Select the variables for the class membership model
Q_VARS= ['KIDSUPTO12Y']

#Select the variables for the latent model
Q_IND = ['CSMEMBER','CARATHOME', 'BIKEATHOME', 'STUDENT', 'RETIRED', 'p_Age']
Indicators = ['pcaratt_1_resp', 'pcaratt_2_resp', 'pcaratt_3_resp', 'pcaratt_4_resp',
               'pcaratt_5_resp', 'pcaratt_6_resp', 'pcaratt_7_resp', 'cs_feel_1_resp',
               'cs_feel_2_resp', 'cs_feel_3_resp', 'cs_feel_4_resp', 'cs_feel_5_resp',
               'cs_feel_6_resp', 'cs_feel_7_resp', 'cs_feel_8_resp', 'cs_feel_9_resp', 'cs_feel_10_resp']

Q_ind = Q[Indicators]

#Input for the class membership
Q_socio1=Q[Q_VARS]
Q_socio = Q_socio1.values

#Input for the indicators
Q_dummies1=Q[Q_IND]
Q_dummies= Q_dummies1.values

#Input for the w
Q_ID = pd.get_dummies(Q['ID']).values

#For the test no individual information enters to the model
Q_ID_test = np.zeros((Q['ID'].shape[0], Q['ID'].shape[0]))


########## Step 5: Output for the indicator (ONLY TRAIN)
I_df= Q[[col for col in Indicators]]
for col in I_df.columns:
    I_df[col]=[float(i)-1 for i in I_df[col].values]
y_ind= I_df.values

### SAME BUT FOR THE TEST
#select the columns
Q = long_data_test[['ID','custom_id','CSMEMBER', 'KIDSUPTO12Y', 'p_Age', 'CARATHOME', 'BIKEATHOME',
                'INCOME_LOW', 'INCOME_MEDIUM', 'INCOME_HIGH',
                'STUDENT', 'EMPLOYED', 'RETIRED', 'WOMAN', 
               'pcaratt_1_resp', 'pcaratt_2_resp', 'pcaratt_3_resp', 'pcaratt_4_resp',
               'pcaratt_5_resp', 'pcaratt_6_resp', 'pcaratt_7_resp', 'cs_feel_1_resp',
               'cs_feel_2_resp', 'cs_feel_3_resp', 'cs_feel_4_resp', 'cs_feel_5_resp',
               'cs_feel_6_resp', 'cs_feel_7_resp', 'cs_feel_8_resp', 'cs_feel_9_resp', 'cs_feel_10_resp']]

#Drop the rows from the same choice
Q.drop_duplicates(subset ="custom_id",keep= 'first', inplace = True)
#Drop the duplicates from the same decision maker
Q.drop_duplicates(subset ="ID",keep= 'first', inplace = True)
Q= Q.drop(columns=['custom_id'])

#Scale age
Q['p_Age'] = preprocessing.scale(Q['p_Age'])

#Select the variables
Q_VARS= ['KIDSUPTO12Y']

Q_IND = ['CSMEMBER','CARATHOME', 'BIKEATHOME', 'STUDENT', 'RETIRED', 'p_Age']
Indicators = ['pcaratt_1_resp', 'pcaratt_2_resp', 'pcaratt_3_resp', 'pcaratt_4_resp',
               'pcaratt_5_resp', 'pcaratt_6_resp', 'pcaratt_7_resp', 'cs_feel_1_resp',
               'cs_feel_2_resp', 'cs_feel_3_resp', 'cs_feel_4_resp', 'cs_feel_5_resp',
               'cs_feel_6_resp', 'cs_feel_7_resp', 'cs_feel_8_resp', 'cs_feel_9_resp', 'cs_feel_10_resp']

#Input for the class membership
Q_socio1_test=Q[Q_VARS]
Q_socio_test = Q_socio1_test.values

#Input for the indicators
Q_dummies1_test=Q[Q_IND]
Q_dummies_test= Q_dummies1_test.values


#### Scale attributes of the alternatives

# scale subscription price and usage cost
long_data['SUB_EURO'] = long_data['SUB_EURO'] / 100
long_data['USECOST_EURO'] = long_data['USECOST_EURO'] * 10

# add a column of ones named 'intercept'.This will be used later to define the Alternative-Specific Constants (ASCs).
long_data['intercept']=np.ones(long_data.shape[0])

### Same but for the test
# scale subscription price and usage cost
long_data_test['SUB_EURO'] = long_data_test['SUB_EURO'] / 100
long_data_test['USECOST_EURO'] = long_data_test['USECOST_EURO'] * 10

# add a column of ones named 'intercept'.This will be used later to define the Alternative-Specific Constants (ASCs).
long_data_test['intercept']=np.ones(long_data_test.shape[0])

if n_classes == 2:
       class_specific_specs = [OrderedDict([('intercept',[1,2,3,4]),
                                          ('USECOST_EURO',[[1,2,3],4]),
                                          ('DAY_PACKAGE',[[1,2,3,4]]),
                                          ('SUB_EURO',[[1,2,3,4]]),
                                          ('WLKTIMEPARK',[[1,2,3,4]]),
                                          ('HOUR_PACKAGE',[[1,2,3,4]]),
                                          (u'ENG_COMB',[[1,2,3,4]]),
                                          (u'PROBCAR',[[1,2,3,4]])]),
                            OrderedDict([('intercept',[1,2,3,4]),
                                          ('USECOST_EURO',[[1,2,3],4]),
                                          ('DAY_PACKAGE',[[1,2,3,4]]),
                                          ('SUB_EURO',[[1,2,3,4]]),
                                          ('WLKTIMEPARK',[[1,2,3,4]]),
                                          ('HOUR_PACKAGE',[[1,2,3,4]]),
                                          (u'ENG_COMB',[[1,2,3,4]]),
                                          (u'PROBCAR',[[1,2,3,4]])])]
       
       class_specific_labels = [OrderedDict([('ASC',['ASC(Roundtrip)',
                                                 'ASC(OW-station)',
                                                 'ASC(OW-freefloating)',
                                                 'ASC(PTP)']),
                                          ('USECOST_EURO',['B_COST_USAGE','B_PTP_COST_USAGE']),
                                          ('DAY_PACKAGE',['B_DAY_PACKAGE']),
                                          ('SUB_EURO',['B_COST_SUBS']),
                                          ('WLKTIMEPARK',['B_WLKTIMEPARK_CPH']),
                                          ('HOUR_PACKAGE',['B_HOUR_PACKAGE']),
                                          (u'ENG_COMB',['B_ENG_COMB']),
                                          (u'PROBCAR',['B_PROBCAR'])]),       
                            OrderedDict([('ASC',['ASC(Roundtrip)',
                                                 'ASC(OW-station)',
                                                 'ASC(OW-freefloating)',
                                                 'ASC(PTP)']),
                                          ('USECOST_EURO',['B_COST_USAGE','B_PTP_COST_USAGE']),
                                          ('DAY_PACKAGE',['B_DAY_PACKAGE']),
                                          ('SUB_EURO',['B_COST_SUBS']),                            
                                          ('WLKTIMEPARK',['B_WLKTIMEPARK_CPH']),
                                          ('HOUR_PACKAGE',['B_HOUR_PACKAGE']),
                                          (u'ENG_COMB',['B_ENG_COMB']),
                                          (u'PROBCAR',['B_PROBCAR'])])]
if n_classes ==3:
       class_specific_specs = [OrderedDict([('intercept',[1,2,3,4]),
                                          ('USECOST_EURO',[[1,2,3],4]),
                                          ('DAY_PACKAGE',[[1,2,3,4]]),
                                          ('SUB_EURO',[[1,2,3,4]]),
                                          ('WLKTIMEPARK',[[1,2,3,4]]),
                                          ('HOUR_PACKAGE',[[1,2,3,4]]),
                                          (u'ENG_COMB',[[1,2,3,4]]),
                                          (u'PROBCAR',[[1,2,3,4]])]),
                            OrderedDict([('intercept',[1,2,3,4]),
                                          ('USECOST_EURO',[[1,2,3],4]),
                                          ('DAY_PACKAGE',[[1,2,3,4]]),
                                          ('SUB_EURO',[[1,2,3,4]]),
                                          ('WLKTIMEPARK',[[1,2,3,4]]),
                                          ('HOUR_PACKAGE',[[1,2,3,4]]),
                                          (u'ENG_COMB',[[1,2,3,4]]),
                                          (u'PROBCAR',[[1,2,3,4]])]),
                            OrderedDict([('intercept',[1,2,3,4]),
                                          ('USECOST_EURO',[[1,2,3],4]),
                                          ('DAY_PACKAGE',[[1,2,3,4]]),
                                          ('SUB_EURO',[[1,2,3,4]]),
                                          ('WLKTIMEPARK',[[1,2,3,4]]),
                                          ('HOUR_PACKAGE',[[1,2,3,4]]),
                                          (u'ENG_COMB',[[1,2,3,4]]),
                                          (u'PROBCAR',[[1,2,3,4]])])]
       
       class_specific_labels = [OrderedDict([('ASC',['ASC(Roundtrip)',
                                                 'ASC(OW-station)',
                                                 'ASC(OW-freefloating)',
                                                 'ASC(PTP)']),
                                          ('USECOST_EURO',['B_COST_USAGE','B_PTP_COST_USAGE']),
                                          ('DAY_PACKAGE',['B_DAY_PACKAGE']),
                                          ('SUB_EURO',['B_COST_SUBS']),
                                          ('WLKTIMEPARK',['B_WLKTIMEPARK_CPH']),
                                          ('HOUR_PACKAGE',['B_HOUR_PACKAGE']),
                                          (u'ENG_COMB',['B_ENG_COMB']),
                                          (u'PROBCAR',['B_PROBCAR'])]),
                            OrderedDict([('ASC',['ASC(Roundtrip)',
                                                 'ASC(OW-station)',
                                                 'ASC(OW-freefloating)',
                                                 'ASC(PTP)']),
                                          ('USECOST_EURO',['B_COST_USAGE','B_PTP_COST_USAGE']),
                                          ('DAY_PACKAGE',['B_DAY_PACKAGE']),
                                          ('SUB_EURO',['B_COST_SUBS']),
                                          ('WLKTIMEPARK',['B_WLKTIMEPARK_CPH']),
                                          ('HOUR_PACKAGE',['B_HOUR_PACKAGE']),
                                          (u'ENG_COMB',['B_ENG_COMB']),
                                          (u'PROBCAR',['B_PROBCAR'])]),          
                            OrderedDict([('ASC',['ASC(Roundtrip)',
                                                 'ASC(OW-station)',
                                                 'ASC(OW-freefloating)',
                                                 'ASC(PTP)']),
                                          ('USECOST_EURO',['B_COST_USAGE','B_PTP_COST_USAGE']),
                                          ('DAY_PACKAGE',['B_DAY_PACKAGE']),
                                          ('SUB_EURO',['B_COST_SUBS']),                            
                                          ('WLKTIMEPARK',['B_WLKTIMEPARK_CPH']),
                                          ('HOUR_PACKAGE',['B_HOUR_PACKAGE']),
                                          (u'ENG_COMB',['B_ENG_COMB']),
                                          (u'PROBCAR',['B_PROBCAR'])])]


# Starting values for the Class-Specific Choice Model parameters
paramClassSpec = []
paramClassSpec.append(np.array([ -3.556395,-2.602622,-2.682666,0.108800,-0.160387,-5.585067, -2.136692, -1.710608,0.030276,-1.225292, -0.147581, 1.348977])) 
paramClassSpec.append(np.array([ 0.695702, 0.672063,-0.238395, 2.140818,-0.153274, -3.659679, -0.967388,  -0.323645,  -0.009497, -0.311659, -0.381454,1.644440])) # for the second class
paramClassSpec.append(np.array([4.898652, 4.920137,  5.595156 , 6.146323,0.059676,-1.165436,-0.051435,-1.240863,-0.042484, 0.063779,-0.311029,0.075516])) # for the third class


###### Initialization of the class membership
#seed(0)
#Random way of initializing the class membership
def initweights(Q_train, nClasses):
    init=[]
    for i in range (Q_train.shape[0]):
        k=np.random.dirichlet(np.ones(nClasses),size=1)
        init.append(k)
    init=np.array(init)
    init=np.reshape(init, (Q_train.shape[0], nClasses))
    return init

IniClassMemRan= initweights(Q_socio, n_classes)



# Train/Estimate the model
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    
    
    emb_model, output= FINAL_test_w_first_copy.lccm_fit(data = long_data,    
                  Q_train = Q_socio,
                  Q_vars = Q_VARS,
                  Q_ID= Q_ID,
                  Q_ID_test = Q_ID_test,
                  n_nodes=n_nodes, 
                  n_latent_vars= n_latent_variables,
                  ind=y_ind,
                  Q_dummies = Q_dummies,
                  N_EPOCHS = n_epochs,
                  ind_id_col = 'ID',
                  obs_id_col = 'custom_id',
                  alt_id_col = 'mode_id',
                  choice_col = 'Choice', 
                  n_classes = n_classes,
                  IniClassMem = IniClassMemRan,
                  class_specific_specs = class_specific_specs,
                  class_specific_labels = class_specific_labels,
                  iter= ITER,
                  Q_test= Q_socio_test,
                  data_test= long_data_test,
                  Q_dummies_test= Q_dummies_test,
                  #outputFilePath = outputFilePath,
                  outputFileName = filename,
                  #paramClassSpec = paramClassSpec
                  )


#### Some additional plots to represent the classes and the latent variables


if n_classes==3:
    prob = pd.read_csv('SampleEnum/'+filename + 'SampleEnum.csv', usecols=[0,1,2,3], names= ['ID', '1', '2','3'])
if n_classes==2:
    prob = pd.read_csv('SampleEnum/'+filename + 'SampleEnum.csv', usecols=[0,1,2], names= ['ID', '1', '2'])

#Prepare sociocharacteristics to concat
Q_dummies= Q_dummies1.reset_index()
Q_dummies= Q_dummies.drop(columns=['index'])
Q_socio= Q_socio1.reset_index()
Q_socio= Q_socio.drop(columns=['index'])

#Concat
data1 = pd.concat([prob, Q_dummies, Q_socio], axis=1)

for i in range(1,n_classes+1):
    for col in Q_IND:
        data1['Prob_'+ str(i) +'_' + col]= data1[str(i)]*data1[col]
    for col in Q_VARS:
        data1['Prob_'+ str(i) +'_' + col]= data1[str(i)]*data1[col]

df_train= pd.DataFrame()

for i in range(1,n_classes+1):
    dict= pd.DataFrame({'CSMEMBER': (data1['Prob_'+ str(i) +'_CSMEMBER'].sum()/np.count_nonzero(data1['CSMEMBER'])*data1['CSMEMBER'].mean())/data1[str(i)].mean(), 'CARATHOME': (data1['Prob_'+ str(i) +'_CARATHOME'].sum()/np.count_nonzero(data1['CARATHOME'])*data1['CARATHOME'].mean())/data1[str(i)].mean(),'BIKEATHOME':(data1['Prob_'+ str(i) +'_BIKEATHOME'].sum()/np.count_nonzero(data1['BIKEATHOME'])*data1['BIKEATHOME'].mean())/data1[str(i)].mean(),
    'KIDSUPTO12Y':(data1['Prob_'+ str(i) +'_KIDSUPTO12Y'].sum()/np.count_nonzero(data1['KIDSUPTO12Y'])*data1['KIDSUPTO12Y'].mean())/data1[str(i)].mean(), 'STUDENT':(data1['Prob_'+ str(i) +'_STUDENT'].sum()/np.count_nonzero(data1['STUDENT'])*data1['STUDENT'].mean())/data1[str(i)].mean(), 'RETIRED':data1['Prob_'+ str(i) +'_RETIRED'].mean()/data1[str(i)].mean()}, index=['Class'+str(i)])
    df_train = pd.concat([df_train, dict])
df_train= df_train.T
df_train.head()

#Plot and save image
ax1= df_train.plot.bar()
plt.ylabel('Percentage of the class population')
plt.title('Representation of the classes')
plt.savefig(filename+'Classes.png')
plt.clf()

#Write on the param file
f= open( filename + 'Param.txt','a') 
f.write("\nVariables used to predict the indicators: ")
f.write(' '.join(str(col) for col in Q_IND))


#### Plot rs
with open('Rs/Rs'+ filename +'.pickle', 'rb') as handle:
    Rs = pickle.load(handle)

#Prepare socio characteristics to concat
Q_socio= Q_socio1.reset_index()
Q_socio= Q_socio.drop(columns=['index'])

#Prepare rs to concat
Rs1=Rs[0].reshape([Q_socio.shape[0],n_latent_variables])
df_Rs =pd.DataFrame(Rs1,columns=['R'+str(i+1) for i in range(n_latent_variables)])

#Concat
result = pd.concat([df_Rs, Q_socio], axis=1)

if (n_latent_variables==2) & len(Q_VARS) > 1:

#Plot the rs with the socio characteristics
    fig, ax = plt.subplots(len(Q_VARS), 1, figsize=(10, 15))

    for i in range(len(Q_VARS)):

        if (i == len(Q_VARS)-1) & Q_VARS.count("p_Age") > 0:
            ax[len(Q_VARS)-1].scatter(result.R1,result.R2, c=result.p_Age, cmap=cm.coolwarm, label='Age')
            ax[len(Q_VARS)-1].legend()
            ax[len(Q_VARS)-1].set_xlabel('R1')
            ax[len(Q_VARS)-1].set_ylabel('R2')

        ax[i].scatter(result.R1[result[Q_VARS[i]]==0],result.R2[result[Q_VARS[i]]==0], color='blue', label='No '+ Q_VARS[i], alpha=0.3)
        ax[i].scatter(result.R1[result[Q_VARS[i]]==1],result.R2[result[Q_VARS[i]]==1], color='red', label=Q_VARS[i], alpha=0.3)
        ax[i].legend()
        ax[i].set_xlabel('R1')
        ax[i].set_ylabel('R2')

    # Save the full figure
    fig.tight_layout()
    fig.savefig(filename + 'Socio.png')


if (n_latent_variables==3) & len(Q_VARS) > 1:

    #Plot the rs with the socio characteristics
    fig, ax = plt.subplots(len(Q_VARS), 3, figsize=(15, 20))

    for i in range(len(Q_VARS)):
    
        if (i== len(Q_VARS)-1) & Q_VARS.count("p_Age") > 0:
            ax[len(Q_VARS)-1,0].scatter(result.R1,result.R2, c=result.p_Age, cmap=cm.coolwarm, label='Age')
            ax[len(Q_VARS)-1,0].legend()
            ax[len(Q_VARS)-1,0].set_xlabel('R1')
            ax[len(Q_VARS)-1,0].set_ylabel('R2')
            ax[len(Q_VARS)-1,1].scatter(result.R1,result.R3, c=result.p_Age, cmap=cm.coolwarm, label='Age')
            ax[len(Q_VARS)-1,1].legend()
            ax[len(Q_VARS)-1,1].set_xlabel('R1')
            ax[len(Q_VARS)-1,1].set_ylabel('R3')
            ax[len(Q_VARS)-1,2].scatter(result.R2,result.R3, c=result.p_Age, cmap=cm.coolwarm, label='Age')
            ax[len(Q_VARS)-1,2].legend()
            ax[len(Q_VARS)-1,2].set_xlabel('R2')
            ax[len(Q_VARS)-1,2].set_ylabel('R3')
    
        ax[i,0].scatter(result.R1[result[Q_VARS[i]]==0],result.R2[result[Q_VARS[i]]==0], color='blue', label='No '+ Q_VARS[i], alpha=0.3)
        ax[i,0].scatter(result.R1[result[Q_VARS[i]]==1],result.R2[result[Q_VARS[i]]==1], color='red', label=Q_VARS[i], alpha=0.3)
        ax[i,0].legend()
        ax[i,0].set_xlabel('R1')
        ax[i,0].set_ylabel('R2')
        ax[i,1].scatter(result.R1[result[Q_VARS[i]]==0],result.R3[result[Q_VARS[i]]==0], color='blue', label='No '+ Q_VARS[i], alpha=0.3)
        ax[i,1].scatter(result.R1[result[Q_VARS[i]]==1],result.R3[result[Q_VARS[i]]==1], color='red', label=Q_VARS[i], alpha=0.3)
        ax[i,1].legend()
        ax[i,1].set_xlabel('R1')
        ax[i,1].set_ylabel('R3')
        ax[i,2].scatter(result.R2[result[Q_VARS[i]]==0],result.R3[result[Q_VARS[i]]==0], color='blue', label='No '+ Q_VARS[i], alpha=0.3)
        ax[i,2].scatter(result.R2[result[Q_VARS[i]]==1],result.R3[result[Q_VARS[i]]==1], color='red', label=Q_VARS[i], alpha=0.3)
        ax[i,2].legend()
        ax[i,2].set_xlabel('R2')
        ax[i,2].set_ylabel('R3')
    
    # Save the full figure
    fig.tight_layout()
    fig.savefig(filename + 'Socio.png')

#### Plot with the indicators

#Prepare sociocharacteristics to concat
Q_dummies= Q_dummies1.reset_index()
Q_dummies= Q_dummies.drop(columns=['index'])

#Concat
result = pd.concat([df_Rs, Q_dummies], axis=1)

if n_latent_variables==2:

#Plot the rs with the socio characteristics
    fig, ax = plt.subplots(len(Q_IND), 1, figsize=(10, 20))
    fig.suptitle('Variables for the class membership')

    for i in range(len(Q_IND)-1):
        ax[i].scatter(result.R1[result[Q_IND[i]]==0],result.R2[result[Q_IND[i]]==0], color='blue', label='No '+ Q_IND[i], alpha=0.3)
        ax[i].scatter(result.R1[result[Q_IND[i]]==1],result.R2[result[Q_IND[i]]==1], color='red', label=Q_IND[i], alpha=0.3)
        ax[i].legend()
        ax[i].set_xlabel('R1')
        ax[i].set_ylabel('R2')

        if Q_IND.count("p_Age") > 0:
            ax[len(Q_IND)-1].scatter(result.R1,result.R2, c=result.p_Age, cmap=cm.coolwarm, label='Age')
            ax[len(Q_IND)-1].legend()
            ax[len(Q_IND)-1].set_xlabel('R1')
            ax[len(Q_IND)-1].set_ylabel('R2')
        
        else:
            ax[len(Q_IND)-1].scatter(result.R1[result[len(Q_IND)-1]==0],result.R2[result[len(Q_IND)-1]==0], color='blue', label='No '+ Q_IND[len(Q_IND)-1], alpha=0.3)
            ax[len(Q_IND)-1].scatter(result.R1[result[len(Q_IND)-1]==1],result.R2[result[len(Q_IND)-1]==1], color='red', label=Q_IND[len(Q_IND)-1], alpha=0.3)
            ax[len(Q_IND)-1].legend()
            ax[len(Q_IND)-1].set_xlabel('R1')
            ax[len(Q_IND)-1].set_ylabel('R2')

if n_latent_variables==3:

    #Plot the rs with the socio characteristics
    fig, ax = plt.subplots(len(Q_IND), 3, figsize=(15, 20))
    fig.suptitle('Variables to predict the indicators:')

    for i in range(len(Q_IND)-1):

        if (i== len(Q_VARS)-1) & Q_VARS.count("p_Age") > 0:
            ax[len(Q_IND)-1,0].scatter(result.R1,result.R2, c=result.p_Age, cmap=cm.coolwarm, label='Age')
            ax[len(Q_IND)-1,0].legend()
            ax[len(Q_IND)-1,0].set_xlabel('R1')
            ax[len(Q_IND)-1,0].set_ylabel('R2')
            ax[len(Q_IND)-1,1].scatter(result.R1,result.R3, c=result.p_Age, cmap=cm.coolwarm, label='Age')
            ax[len(Q_IND)-1,1].legend()
            ax[len(Q_IND)-1,1].set_xlabel('R1')
            ax[len(Q_IND)-1,1].set_ylabel('R3')
            ax[len(Q_IND)-1,2].scatter(result.R2,result.R3, c=result.p_Age, cmap=cm.coolwarm, label='Age')
            ax[len(Q_IND)-1,2].legend()
            ax[len(Q_IND)-1,2].set_xlabel('R2')
            ax[len(Q_IND)-1,2].set_ylabel('R3')

        else:
            ax[i,0].scatter(result.R1[result[Q_IND[i]]==0],result.R2[result[Q_IND[i]]==0], color='blue', label='No '+ Q_IND[i], alpha=0.3)
            ax[i,0].scatter(result.R1[result[Q_IND[i]]==1],result.R2[result[Q_IND[i]]==1], color='red', label=Q_IND[i], alpha=0.3)
            ax[i,0].legend()
            ax[i,0].set_xlabel('R1')
            ax[i,0].set_ylabel('R2')
            ax[i,1].scatter(result.R1[result[Q_IND[i]]==0],result.R3[result[Q_IND[i]]==0], color='blue', label='No '+ Q_IND[i], alpha=0.3)
            ax[i,1].scatter(result.R1[result[Q_IND[i]]==1],result.R3[result[Q_IND[i]]==1], color='red', label=Q_IND[i], alpha=0.3)
            ax[i,1].legend()
            ax[i,1].set_xlabel('R1')
            ax[i,1].set_ylabel('R3')
            ax[i,2].scatter(result.R2[result[Q_IND[i]]==0],result.R3[result[Q_IND[i]]==0], color='blue', label='No '+ Q_IND[i], alpha=0.3)
            ax[i,2].scatter(result.R2[result[Q_IND[i]]==1],result.R3[result[Q_IND[i]]==1], color='red', label=Q_IND[i], alpha=0.3)
            ax[i,2].legend()
            ax[i,2].set_xlabel('R2')
            ax[i,2].set_ylabel('R3')


# Save the full figure
fig.tight_layout()
fig.savefig(filename + 'Ind.png')

#Get ws
with open('Ws/W'+ filename +'.pickle', 'rb') as handle:
    W = pickle.load(handle)

#Plot the w distribution
plt.figure()
plt.hist(W, bins=80)
plt.xlabel('W')
plt.title('Distribution of w for the sample')
plt.savefig(filename + 'W.png')

#### Plot with the indicators in a color map

#Prepare sociocharacteristics to concat
Q_dummies= Q_ind.reset_index()
Q_dummies= Q_dummies.drop(columns=['index'])

#Concat
result = pd.concat([df_Rs, Q_dummies], axis=1)

if n_latent_variables==2:

#Plot the rs with the socio characteristics
    fig, ax = plt.subplots(round(len(Indicators)/2), 2, figsize=(15, 30))
    count= 0
    for i in range(round(len(Indicators)/2)):
        for j in range(2):
            scatter= ax[i,j].scatter(result.R1,result.R2, c=result[Indicators[count]], cmap=cm.coolwarm, label='Indicator '+str(count), alpha=0.6)
            ax[i,j].legend(*scatter.legend_elements())
            ax[i, j].set_xlabel('R1')
            ax[i, j].set_ylabel('R2')
            ax[i, j].set_title('Indicator '+str(count))
            count += 1

    # Save the full figure
    fig.tight_layout()
    fig.savefig(filename + 'Indicators.png')