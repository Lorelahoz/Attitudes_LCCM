"""
@name:      Latent Class Choice Model with ML for attitudinal statements (EM algorithm)
@author:    Lorena Torres Lahoz
@summary:   Contains functions necessary for estimating latent class choice models with latent variables and attitudinal statements
            using the Expectation Maximization algorithm. This version also accounts
            for choice based sampling using Weighted Exogenous Sample Maximum
            Likelihood.
General References
------------------
This code is based on the latent class choice model (lccm) package which can be downloaded from:
    https://github.com/ferasz/LCCM
"""

import numpy as np 
from scipy.sparse import coo_matrix
from scipy.optimize import minimize
import scipy.stats
from datetime import datetime
import warnings

from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten, concatenate, Lambda, Concatenate, ZeroPadding1D
from keras.layers import Conv2D, Add, Reshape
import tensorflow as tf

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras import backend as K
from scipy.stats import norm
import pylogit
import pandas as pd
from keras import regularizers

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Ordinal Output layer
class OrdinalOutput(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.t0 = K.constant(-np.inf, shape=(1,1))
        self.tK = K.constant(np.inf, shape=(1,1))
        super(OrdinalOutput, self).__init__(**kwargs)

    def build(self, input_shape):
        self.thresholds = self.add_weight(
            name='thresholds',
            shape=(input_shape[1], self.output_dim - 1),
            initializer=self.sorted_initializer('glorot_uniform'),
            trainable=True)
        self.thresholds = K.concatenate(
            [self.t0, self.thresholds, self.tK],
            axis=-1)
        super(OrdinalOutput, self).build(input_shape)

    def call(self, x):
        output = (
            K.sigmoid(self.thresholds[:, 1:] - x) - 
            K.sigmoid(self.thresholds[:, :-1] - x))
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def sorted_initializer(self, initializer):
        # Returns a function that returns a sorted
        # initialization based on an initializer string
        def sorter(shape, dtype=None):
            # Returns a sorted initialization
            init_func = initializers.get(initializer)
            init = K.eval(init_func(shape, None))
            init = np.sort(init)
            init = K.cast_to_floatx(init)
            return init
        return sorter


#Model architecture
def Latent_Class_Model(nIDs, indicators_vars=6, socio_num= 5, n_nodes=6, n_latent_vars=2,
                       classes_num=3, logits_activation= 'softmax', n_ind=17):


    #Input the sociocharacteristics for the class membership
    sc_input1 = Input(shape= (socio_num,), name= 'input_sc')
    #Reshape the inputs in [n_sc, 1, 1]
    r_inputR= Reshape([socio_num,1,1], name= 'reshape_dummies')(sc_input1)


    #The input socio characteristics for the indicators
    r_input= Input(shape= (indicators_vars,), name= 'input_dummies')
    #Apply dense layer with relu activation
    dense= Dense(units= n_nodes, activation='relu', name= 'Dense_NN_per_frame1', kernel_regularizer=regularizers.l2(0.01))(r_input)
    #Add dropout
    emb_dropout= Dropout(0.2, name= 'dropout_layer')(dense)
    #Apply a dense layer with bias to each latent variable (different weight for each latent variable)
    new_features= Dense(units= n_latent_vars, use_bias=False, name= 'Output_new_feature')(emb_dropout)
    #Reshape the latent variables in [1,latent]
    new_featuresR= Reshape([1,n_latent_vars], name= 'Remove_Dim')(new_features)

    #Input the IDs
    ID_input = Input(shape= (nIDs,), name= 'input_ID')
    #Apply a dense linear layer
    denseID= Dense(units= 1, name= 'Linear_IDs', use_bias=False, kernel_regularizer=regularizers.l2(0.01))(ID_input)

    all_sc=[]

    for i in range(socio_num):
    
        r= Lambda(lambda z: z[:,i,:,:], name= 'get_sc_'+str(i+1))(r_inputR)
        r= Reshape([1,1], name= 'reshape_sc_'+str(i+1))(r)
        rs_padded_classes=[]
        for class_ in range(classes_num-1):
            padded_r= ZeroPadding1D(padding=[class_,classes_num-class_-1],
                                   name='pad_sc'+str(i+1)+'_class'+str(class_+1))(r)
            rs_padded_classes.append(padded_r)
     
        if len(rs_padded_classes)>1:
            r_for_all_classes = Concatenate(name= 'Concat_sc_'+str(i+1), axis=1)(rs_padded_classes)
        else:
            r_for_all_classes= padded_r
        r_for_all_classes = Reshape([classes_num-1, classes_num,1],name= 'Reshape_concat_sc_'+str(i+1))(r_for_all_classes)
        all_sc.append(r_for_all_classes)
    
    all_rs=[]

    for i in range(n_latent_vars):
    
        r= Lambda(lambda z: z[:,:,i], name= 'get_r_'+str(i+1))(new_featuresR)
        r= Reshape([1,1], name= 'reshape_r_'+str(i+1))(r)
        rs_padded_classes=[]
        for class_ in range(classes_num-1):
            padded_r= ZeroPadding1D(padding=[class_,classes_num-class_-1],
                                   name='pad_r'+str(i+1)+'_class'+str(class_+1))(r)
            rs_padded_classes.append(padded_r)
     
        if len(rs_padded_classes)>1:
            r_for_all_classes = Concatenate(name= 'Concat_r_'+str(i+1), axis=1)(rs_padded_classes)
        else:
            r_for_all_classes= padded_r
        print(r_for_all_classes)
        r_for_all_classes = Reshape([classes_num-1, classes_num,1],name= 'Reshape_concat_r_'+str(i+1))(r_for_all_classes)
        all_rs.append(r_for_all_classes)
        
    
    r= Reshape([1,1], name= 'reshape_ID')(denseID)
    rs_padded_classes=[]
    for class_ in range(classes_num-1):
        padded_r= ZeroPadding1D(padding=[class_,classes_num-class_-1],
                                   name='pad_r_ID_class'+str(class_+1))(r)
        rs_padded_classes.append(padded_r)
        print(padded_r)
     
    if len(rs_padded_classes)>1:
            r_for_all_classes = Concatenate(name= 'Concat_r_ID', axis=1)(rs_padded_classes)
    else:
            r_for_all_classes= padded_r

    print(r_for_all_classes)
    r_for_all_classes = Reshape([classes_num-1, classes_num,1],name= 'Reshape_concat_r_ID')(r_for_all_classes)
    all_rs.append(r_for_all_classes)

    #ASC input for the classes    
    ascs_classes_input= Input(shape= (classes_num-1,classes_num,1), name= 'ascs_for_classes')

    #Concatenate all the variables that go into the utility
    ascs_and_embs= Concatenate(axis=1 , name='concat_ascs_and_embs')([ascs_classes_input]+all_sc+all_rs)
    print(ascs_and_embs)
    print(classes_num-1+socio_num*(classes_num-1)+(n_latent_vars*(classes_num-1))+1)
    #Apply a convolution layer to all the inputs
    utilities_classes= Conv2D(filters= 1, kernel_size= [classes_num-1+socio_num*(classes_num-1)+(n_latent_vars*(classes_num-1))+1*(classes_num-1),1],
                        strides= (1,1), padding= 'valid', name= 'Utils_Classes',
                        use_bias= False, trainable= True)(ascs_and_embs)
    
    print(utilities_classes)
    #Reshape utilitis in size [num_classes]
    utilities_classesR= Reshape([classes_num], name= 'Flatten_Utils_Classes')(utilities_classes)
    #Apply logit activation
    logits=  Activation(logits_activation, name= 'Class')(utilities_classesR)
    #create the list with all the outputs
    output_acts=[logits]
    

    r_ID= Concatenate(axis=1, name='concat_rs_and_ids')([new_features] + [denseID])
    print(r_ID)

    #For each latent variable
    for i in range(n_ind):
        #Make a linear combination with intercept for each question with a Dense layer
        output_act = Dense(1, activation='linear', use_bias=False, name="Output_linear_act"+ str(i+1), kernel_regularizer=regularizers.l2(0.01))(r_ID)
        #Convert each indicator's utility in an OrdinalOutput
        output_act= OrdinalOutput(5, name="Output_ord_act"+ str(i+1))(output_act)
        #Append it to the list of outputs
        output_acts.append(output_act)
    
    #Create the model with the inputs and outputs defined
    model= Model(inputs= [ascs_classes_input, r_input, sc_input1, ID_input], outputs= output_acts, name= 'Classes')

    print(model.summary())
    
    return model


def get_inverse_Hessian(model, model_inputs, labels, layer_name='Utilities'):

    """ This function was copied from: https://github.com/BSifringer/EnhancedDCM
    and was modified to handle singular matrix cases."""

    data_size = len(model_inputs[0])

# Get layer and gradient w.r.t. loss
    beta_layer = model.get_layer(layer_name)
    beta_gradient = K.gradients(model.total_loss, beta_layer.weights[0])[0]

# Get second order derivative operators (linewise of Hessian)
    Hessian_lines_op = {}
    for i in range(len(beta_layer.get_weights()[0])):
        Hessian_lines_op[i] = K.gradients(beta_gradient[i], beta_layer.weights[0])
    input_tensors= model.inputs + [model.sample_weights[0]] + [model.targets[0]] + [K.learning_phase()]
    get_Hess_funcs = {}
    for i in range(len(Hessian_lines_op)):
        get_Hess_funcs[i] = K.function(inputs=input_tensors, outputs=Hessian_lines_op[i])

# Line by line Hessian average multiplied by data length (due to automatic normalization)
    Hessian=[]
    func_inputs=[*[inputs for inputs in model_inputs], np.ones(data_size), labels[0], 0]
        
    for j in range(len(Hessian_lines_op)):
        Hessian.append((np.array(get_Hess_funcs[j](func_inputs))))
    Hessian = np.squeeze(Hessian)*data_size

# The inverse Hessian:
    try:
        invHess = np.linalg.inv(Hessian)
    except np.linalg.LinAlgError:
        print(LinAlgError('Singular matrix'))
        return np.nan

    return invHess

def get_stds(model, model_inputs, labels, layer_name='Utilities'):

    """ Gets the diagonal of the inverse Hessian, square rooted
        This function was copied from: https://github.com/BSifringer/EnhancedDCM
        and was modified to handle singular matrix cases."""

    inv_Hess = get_inverse_Hessian(model, model_inputs, labels, layer_name)

    if isinstance(inv_Hess, float):

        return np.nan

    else:

        stds = [inv_Hess[i][i]**0.5 for i in range(inv_Hess.shape[0])]

        return np.array(stds).flatten()
    

def model_summary(trained_model, Q_train, y_train, ind, n_latent_vars,  outputFileName, Q_vars_names):


    betas_embs= trained_model.get_layer('Utils_Classes').get_weights()[0].reshape(-1)
    
    emb_betas_stds= get_stds(trained_model, Q_train, y_train, layer_name='Utils_Classes')
    

    if not isinstance(emb_betas_stds, float):

        betas= np.empty([ind.shape[1], n_latent_vars+1])
        index1= ['Ind ' + str(i) for i in range(ind.shape[1])]
        #stds=[]
        for i in range(ind.shape[1]):
            #Returns the kernel matrix and the bias vector
            for j in range(n_latent_vars+1):
            #bias.append(float((trained_model.get_layer("Output_linear_act"+ str(i+1)).get_weights()[1].reshape(-1)).astype(str)))
                betas[i][j] = float(trained_model.get_layer("Output_linear_act"+ str(i+1)).get_weights()[0][j])
            #stds.append(get_stds(trained_model, Q_train, y_train, layer_name="Output_linear_act"+ str(i+1)))

        inds=pd.DataFrame(index=index1,
                      data=betas,
                      columns=['Beta'+str(i+1) for i in range(n_latent_vars)]+ ['Beta_w'])

        z_embs= betas_embs/emb_betas_stds

        p_embs = (1-norm.cdf(abs(z_embs)))*2
       
        stats_embs=np.array(list(zip(Q_vars_names, betas_embs, emb_betas_stds, z_embs, p_embs)))
        
        stats_all= stats_embs

        #Get the ws
        ws= trained_model.get_layer('Linear_IDs').get_weights()[0].reshape(-1)

        with open('Ws/W'+ outputFileName +'.pickle', 'wb') as handle:
            pickle.dump(ws, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        df_stats=pd.DataFrame(index=[i[0] for i in stats_all],
                          data=np.array([[np.float(i[1]) for i in stats_all],[np.float(i[2]) for i in stats_all],
                                         [np.float(i[3]) for i in stats_all],
                                         [np.round(np.float(i[4]),4) for i in stats_all]]).T,
                          columns=['Betas','St errors', 't-stat','p-value'])

        return df_stats, inds

    else:
   
        return np.nan


# Global variables
emTol = 0.05
llTol = 1e-06
grTol = 1e-06
maxIters = 10000

def processClassSpecificPanel(dms, dmID, obsID, altID, choice):
    """
    Method that constructs a tuple and three sparse matrices containing information 
    on available observations, and available and chosen alternative
    
    Parameters
    ----------
    dms : 1D numpy array of size nDms.
        Each element identifies a unique decision-maker in the dataset.
    dmID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which decision-maker.
    obsID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which observation.
    altID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which alternative.
    choice : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to alternatives that were 
        chosen during the corresponding observation.
    
    Returns
    -------
    altAvTuple : a tuple containing arrays of size nRows.
        The first array denotes which row in the data file corresponds to which 
        row in the data file (redundant but retained for conceptual elegance) and 
        the second array denotes which row in the data file corresponds to which 
        observation in the data file.    
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element of the returned matrix  is 1 if the alternative corresponding 
        to the ith row in the data file was chosen during observation j, and 0 otherwise.   
    obsAv : sprase matrix of size (nObs x nDms).
        The (i, j)th element of the returned matrix is 1 if observation i corresponds to 
        decision-maker j, and 0 otherwise.
    rowAv : sparse matrix of size (nRows x (nAlts * nDms)).
        The (i, ((n - 1) * nAlts) + j)th element of the returned matrix is 1 if the ith row 
        in the data file corresponds to the jth alternative and the nth decision-maker, 
        and 0 otherwise.   
    """
    
    nRows = choice.shape[0] #number of rows
    alts = np.unique(altID) 
    nAlts = alts.shape[0] #number of alternatives
    obs = np.unique(obsID)
    nObs = obs.shape[0] #number of observations
    nDms = dms.shape[0] #number of decision makers
    
    xAlt, yAlt = np.zeros((nRows)), np.zeros((nRows))
    xChosen, yChosen = np.zeros((nObs)), np.zeros((nObs))
    xObs, yObs = np.zeros((nObs)), np.zeros((nObs))
    xRow, yRow = np.zeros((nRows)), np.zeros((nRows))

    currentRow, currentObs, currentDM = 0, 0, 0    
    for n in dms:
        obs = np.unique(np.extract(dmID == n, obsID)) #observations that belong to each decision maker
        for k in obs:      
            xObs[currentObs], yObs[currentObs] = currentObs, currentDM
            cAlts = np.extract((dmID == n) & (obsID == k), altID)   #las alternativas de cada observacion para cada decision maker     
            for j in cAlts: 
                xAlt[currentRow], yAlt[currentRow] = currentRow, currentObs   
                xRow[currentRow], yRow[currentRow] = currentRow, (np.where(dms == n)[0][0] * nAlts) + np.where(alts == j)[0][0]
                if np.extract((dmID == n) & (obsID == k) & (altID == j), choice) == 1:  #si el decision maker selecciona esa alternativa en ese row, entonces guardo el row y la observation             
                    xChosen[currentObs], yChosen[currentObs] = currentRow, currentObs
                currentRow += 1
            currentObs += 1
        currentDM += 1
            
    altAvTuple = (xAlt, yAlt)
    altChosen = coo_matrix((np.ones((nObs)), (xChosen, yChosen)), shape = (nRows, nObs))
    obsAv = coo_matrix((np.ones((nObs)), (xObs, yObs)), shape = (nObs, nDms))
    rowAv = coo_matrix((np.ones((nRows)), (xRow, yRow)), shape = (nRows, nDms * nAlts))
    
    return altAvTuple, altChosen, obsAv, rowAv
    
    
def imposeCSConstraints(altID, availAlts):
    """
    Method that constrains the choice set for each of the decision-makers across the different
    latent classes following the imposed choice-set by the analyst to each class. 
    Usually, when the data is in longformat, this would not be necessary, since the 
    file would contain rows for only those alternatives that are available. However, 
    in an LCCM, the analyst may wish to impose additional constraints to introduce 
    choice-set heterogeneity.
    
    Parameters
    ----------
    altID : 1D numpy array of size nRows.
        Identifies which rows in the data correspond to which alternative.
    availAlts : List of size nClasses.
        Determines choice set constraints for each of the classes. The sth element is an 
        array containing identifiers for the alternatives that are available to decision-makers 
        belonging to the sth class.
    
    Returns
    -------
    altAvVec : 1D numpy array of size nRows.
        An element is 1 if the alternative corresponding to that row in the data 
        file is available, and 0 otherwise.
    """   
    altAvVec = np.zeros(altID.shape[0]) != 0   
    for availAlt in availAlts:
        altAvVec = altAvVec | (altID == availAlt)
    return altAvVec.astype(int)


def calClassMemProb(Q_train, nClasses, weights, N_EPOCHS,N_current_step, n_nodes, n_latent_vars, ind, Q_dummies, outputFileName, Q_ID, mnl_classes):
    """
    Function that calculates the class membership probabilities for each observation in the
    dataset.
    
    Returns
    -------
    p : 2D numpy array of size 1 x (nDms x nClasses).
        Identifies the class membership probabilities for each individual and 
        each available latent class.
    """
    
    #Define model structure
    optimizer = Adam(clipnorm= 50.)
        
    dict_losses={}
    dict_losses['Class']='categorical_crossentropy'
    history=0

    for i in range(ind.shape[-1]):

        dict_losses["Output_ord_act"+ str(i+1)]= 'sparse_categorical_crossentropy'


    mnl_classes.compile(optimizer= optimizer, metrics= ["accuracy"], loss= dict_losses)
        

    Callback = EarlyStopping(monitor= 'loss', min_delta= 0, patience= 20)
    
    Ascs_Train= np.zeros((len(Q_train),nClasses-1, nClasses,1))
    for i in range(nClasses-1):
        Ascs_Train[:,i,i,:]=1

    #Train the class model with the old posterior
    outputList= [weights]
    
    for i in ind.T:
        outputList.append(np.array(i).T)

    
    history = mnl_classes.fit([Ascs_Train, Q_dummies, Q_train, Q_ID], outputList, epochs= N_EPOCHS, batch_size=32, shuffle= 'batch',
               verbose= 1, callbacks=[Callback])
    
    
    #Get the class probabilities
    predictions= mnl_classes.predict(x= {'ascs_for_classes': Ascs_Train, 'input_dummies':Q_dummies, 'input_sc': Q_train, 'input_ID': Q_ID})
    p=predictions[0]
    #save the model
    N_current_step +=1


    return p, history.history, mnl_classes


def calClassSpecificProbPanel(param, expVars, altAvMat, altChosen, obsAv):
    """
    Function that calculates the class specific probabilities for each decision-maker in the
    dataset for each class
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nRows)).
        Contains explanatory variables.
    altAvMat : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars is available to the decision-maker corresponding to the 
        jth observation, and 0 otherwise.
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars was chosen by the decision-maker corresponding to the 
        jth observation, and 0 otherwise.
    obsAv : sparse matrix of size (nObs x nInds).
        The (i, j)th element equals 1 if the ith observation in the dataset corresponds 
        to the jth decision-maker, and 0 otherwise.
    
    Returns
    -------
    np.exp(lPInd) : 2D numpy array of size 1 x nInds.
        Identifies the class specific probabilities for each individual in the 
        dataset.
    """
    v = np.dot(param[None, :], expVars)       # v is 1 x nRows
    ev = np.exp(v)                            # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                  # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                  # As precaution when exp(v) is too close to zero
    nev = ev * altAvMat                       # nev is 1 x nObs
    nnev = altAvMat * np.transpose(nev)       # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))     # p is 1 x nRows 
    p[np.isinf(p)] = 1e-200                   # When none of the alternatives are available
    pObs = p * altChosen                      # pObs is 1 x nObs
    lPObs = np.log(pObs)                      # lPObs is 1 x nObs
    lPInd = lPObs * obsAv                     # lPInd is 1 x nInds
    return np.exp(lPInd)                      # prob is 1 x nInds

         

def wtLogitPanel(param, expVars, altAv, weightsProb, weightsGr, altChosen, obsAv, choice):
    """
    Function that calculates the log-likelihood function and the gradient for a weighted
    multinomial logit model with panel data. 
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nRows)).
        Contains explanatory variables.
    altAv : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars is available to the decision-maker corresponding to the 
        jth observation, and 0 otherwise. 
    weightsProb : 1D numpy array of size nInds.
        The jth element is the weight to be used for the jth decision-maker.
    weightsGr : 1D numpy array of size nRows.
        The jth element is the weight to be used for the jth row in the dataset. 
        The weights will be the same for all rows in the dataset corresponding to 
        the same decision-maker. However, passing it as a separate parameter speeds up the optimization.   
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith column 
        in expVars was chosen by the decision-maker corresponding to the jth observation,
        and 0 otherwise.
    obsAv : sparse matrix of size (nObs x nInds).
        The (i, j)th element equals 1 if the ith observation in the dataset corresponds 
        to the jth decision-maker, and 0 otherwise.
    choice : 1D numpy array of size nRows.
        The jth element equals 1 if the alternative corresponding to the jth column 
        in expVars was chosen by the decision-maker corresponding to that observation, and 0 otherwise.
        
    Returns
    -------
    ll : a scalar.
        Log-likelihood value for the weighted multinomidal logit model.
    np.asarray(gr).flatten() : 1D numpy array of size nExpVars.
        Gradient for the weighted multinomial logit model.
    
    """       
    v = np.dot(param[None, :], expVars)         # v is 1 x nRows
    ev = np.exp(v)                              # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                    # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                    # As precaution when exp(v) is too close to zero
    nev = ev * altAv                            # nev is 1 x nObs
    nnev = altAv * np.transpose(nev)            # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))       # p is 1 x nRows 
    p[np.isinf(p)] = 1e-200                     # When none of the alternatives are available
    p[p < 1e-200] = 1e-200                      # As precaution when p is too close to zero
    tgr = choice - np.transpose(p)              # ttgr is nRows x 1
    ttgr = -np.multiply(weightsGr, tgr)         # tgr is nRows x 1
    gr = np.dot(expVars, ttgr)                  # gr is nExpVars x 1
    pObs = p * altChosen                        # pObs is 1 x nObs
    lPObs = np.log(pObs)                        # lPObs is 1 x nObs
    lPInd = lPObs * obsAv                       # lPInd is 1 x nInds
    wtLPInd = np.multiply(lPInd, weightsProb)   # wtLPInd is 1 x nInds
    ll = -np.sum(wtLPInd)                       # ll is a scalar
    
    return ll, np.asarray(gr).flatten()
    

def calStdErrWtLogitPanel(param, expVars, altAv, weightsProb, weightsGr, altChosen, obsAv, choice):
    """
    Function that calculates the log-likelihood function and the gradient for a weighted
    multinomial logit model with panel data. 
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nRows)).
        Contains explanatory variables.
    altAv : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars is available to the decision-maker corresponding to the 
        jth observation, and 0 otherwise. 
    weightsProb : 1D numpy array of size nInds.
        The jth element is the weight to be used for the jth decision-maker.
    weightsGr : 1D numpy array of size nRows.
        The jth element is the weight to be used for the jth row in the dataset. 
        The weights will be the same for all rows in the dataset corresponding to 
        the same decision-maker. However, passing it as a separate parameter speeds up the optimization.   
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith column 
        in expVars was chosen by the decision-maker corresponding to the jth observation,
        and 0 otherwise.
    obsAv : sparse matrix of size (nObs x nInds).
        The (i, j)th element equals 1 if the ith observation in the dataset corresponds 
        to the jth decision-maker, and 0 otherwise.
    choice : 1D numpy array of size nRows.
        The jth element equals 1 if the alternative corresponding to the jth column 
        in expVars was chosen by the decision-maker corresponding to that observation, and 0 otherwise.
        
    Returns
    -------
    se : 2D numpy array of size (nExpVars x 1).
        Standard error for the weighted multinomidal logit model.
    
    """ 
    v = np.dot(param[None, :], expVars)         # v is 1 x nRows
    ev = np.exp(v)                              # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                    # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                    # As precaution when exp(v) is too close to zero
    nev = ev * altAv                            # nev is 1 x nObs
    nnev = altAv * np.transpose(nev)            # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))       # p is 1 x nRows 
    p[np.isinf(p)] = 1e-200                     # When none of the alternatives are available
    p[p < 1e-200] = 1e-200                      # As precaution when p is too close to zero
    tgr = choice - np.transpose(p)              # ttgr is nRows x 1
    ttgr = np.multiply(weightsGr, tgr)          # tgr is nRows x 1
    gr = np.tile(ttgr, (1, expVars.shape[0]))   # gr is nRows x nExpVars 
    sgr = np.multiply(np.transpose(expVars),gr) # sgr is nRows x nExpVars 
    hess = np.dot(np.transpose(sgr), sgr)       # hess is nExpVars x nExpVars 
    try:                                        # iHess is nExpVars x nExpVars 
        iHess = np.linalg.inv(hess)             # If hess is non-singular
    except:
        iHess = np.identity(expVars.shape[0])   # If hess is singular
    se = np.sqrt(np.diagonal(iHess))            # se is nExpVars x 1

    return se
    

def displayOutput(outputFile, startTime, llEstimation,llNull,
                  nClasses, Q_vars, nchoices,
        namesExpVarsClassSpec, paramClassSpec, stdErrClassSpec, obsID, n_latent_vars, n_nodes, outputFilePath, outputFileName, iter, nInds, num_vars_indicators): 
    """
    Function that displays the estimation results and model's stastical fit results. 
    
    """
    
    num_class_specific_model = 0
    for i in range(0, nClasses):
        num_class_specific_model = num_class_specific_model + paramClassSpec[i].shape[0]
    num_parameters_total = num_class_specific_model + n_nodes*(num_vars_indicators+1) + n_nodes*(n_latent_vars+1) + ((n_latent_vars+len(Q_vars)+1)*(nClasses-1))+ (nClasses-1) #parameters of the two dense layer + parameters of the class membership + betas for the ws
    #for the class specific + for the densely connected layer + for the utility of the classes
    rho_squared = 1 - llEstimation/llNull
    rho_bar_squared = 1 - (llEstimation-num_parameters_total)/llNull
    AIC = -2*llEstimation + 2*num_parameters_total
    BIC = -2*llEstimation  + num_parameters_total*np.log(np.unique(obsID).shape[0])
    timeElapsed = datetime.now() - startTime
    timeElapsed = (timeElapsed.days * 24.0 * 60.0) + (timeElapsed.seconds/60.0)
    
    # Display screen
    
    print ("\n")
    print ("Number of Parameters:".ljust(45,' '), (str(num_parameters_total).rjust(10,' ')))
    print ("Number of Observations:".ljust(45, ' '),(str(np.unique(obsID).shape[0]).rjust(10,' ')))   
    print ("Null Log-Likelihood:".ljust(45, ' '),(str(round(llNull,2)).rjust(10,' ')))  
    print ("Fitted Log-Likelihood:".ljust(45, ' '),(str(round(llEstimation,2)).rjust(10,' ')))   
    print ("Rho-Squared:".ljust(45, ' '),(str(round(rho_squared,2)).rjust(10,' '))) 
    print ("Rho-Bar-Squared:".ljust(45, ' '),(str(round(rho_bar_squared,2)).rjust(10,' ')))
    print ("AIC:".ljust(45, ' '),(str(round(AIC,2)).rjust(10,' '))) 
    print ("BIC:".ljust(45, ' '),(str(round(BIC)).rjust(10,' '))) 
    print ("Estimation time (minutes):".ljust(45, ' '),(str(round(timeElapsed,2)).rjust(10,' '))) 
    print ("\n")

    #Write on a file
    open(outputFilePath + outputFileName + 'Param.txt', 'w').close()
    f= open(outputFilePath + outputFileName + 'Param.txt','a')
    f.write("Number of Parameters:".ljust(45,' ')+ (str(num_parameters_total).rjust(10,' '))+'\n')
    f.write("Number of Observations in the train:".ljust(45, ' ')+(str(np.unique(obsID).shape[0]).rjust(10,' '))+'\n')
    f.write("Number of individuals in the train:".ljust(45, ' ')+(str(nInds)).rjust(10,' ')+'\n') 
    f.write("Null Log-Likelihood:".ljust(45, ' ')+(str(round(llNull,2)).rjust(10,' '))+'\n')
    f.write(("Fitted train Log-Likelihood:".ljust(45, ' ')+(str(round(llEstimation,2)).rjust(10,' ')))+'\n')
    f.write(("Rho-Squared:".ljust(45, ' ')+(str(round(rho_squared,2)).rjust(10,' ')))+'\n')
    f.write("Rho-Bar-Squared:".ljust(45, ' ')+(str(round(rho_bar_squared,2)).rjust(10,' '))+'\n')
    f.write("AIC:".ljust(45, ' ')+(str(round(AIC,2)).rjust(10,' '))+'\n') 
    f.write("BIC:".ljust(45, ' ')+(str(round(BIC)).rjust(10,' '))+'\n')
    f.write("Estimation time (minutes):".ljust(45, ' ')+(str(round(timeElapsed,2)).rjust(10,' '))+'\n')
    f.write("Number of iterations".ljust(45, ' ')+(str(iter))+'\n')
    f.close()

    for s in range(0, nClasses):
        print
        print('Class',s + 1,' Model: ')
        print('-----------------------------------------------------------------------------------------')
        print('Variables                                     betas    std_err     t_stat    p_value')
        print('-----------------------------------------------------------------------------------------')
        for k in range(0, len(namesExpVarsClassSpec[s])):
            print('%-45s %10.4f %10.4f %10.4f %10.4f' %(namesExpVarsClassSpec[s][k], paramClassSpec[s][k], 
                    stdErrClassSpec[s][k], paramClassSpec[s][k]/stdErrClassSpec[s][k], scipy.stats.norm.sf(abs(paramClassSpec[s][k]/stdErrClassSpec[s][k]))*2))
        print('-----------------------------------------------------------------------------------------')

    print('Class Membership Model:')
    print('-----------------------------------------------------------------------------------------')

    #Write on a file
    for s in range(0, nClasses):
        stats_all=np.array(list(zip(namesExpVarsClassSpec[s], paramClassSpec[s],stdErrClassSpec[s],paramClassSpec[s]/stdErrClassSpec[s], scipy.stats.norm.sf(abs(paramClassSpec[s]/stdErrClassSpec[s]))*2)))
        
        
        df_stats=pd.DataFrame(index=[i[0] for i in stats_all],
                          data=np.array([[np.float(i[1]) for i in stats_all],[np.float(i[2]) for i in stats_all],
                                         [np.float(i[3]) for i in stats_all],
                                         [np.round(np.float(i[4]),4) for i in stats_all]]).T,
                          columns=['Betas','St errors', 't-stat','p-value'])

        f= open(outputFilePath + outputFileName + 'Param.txt','a')
        f.write('\n')
        f.write('Choice model parameters class:'+str(s+1)+'\n')
        f.write(df_stats.to_string())
        f.write('\n')
        f.close()    
    return num_parameters_total

def processData(inds, indID, nClasses, #expVarsClassMem,
                availIndClasses, obsID, altID, choice, availAlts):
    """
    Function that takes the raw data and processes it to construct arrays and matrices that
    are subsequently used during estimation. 
    
    Parameters
    ----------
    inds : 1D numpy array of size nInds (total number of individuals in the dataset).
        Depicts total number of decision-makers in the dataset.    
    indID : 1D numpy array of size nRows.
        The jth element identifies the decision-maker corresponding to the jth row
        in the dataset.
    nClasses : Integer.
        Number of classes to be estimated by the model.
    obsID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which observation.
    altID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which alternative.
    choice : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to alternatives that were 
        chosen during the corresponding observation.
    availAlts : List of size nClasses.
        Determines choice set constraints for each of the classes. The sth element is an 
        array containing identifiers for the alternatives that are available to decision-makers 
        belonging to the sth class.
        
    Returns
    -------
    nInds : Integer.
        Total number of individuals/decision-makers in the dataset. 
    altAv : List of size nClasses. 
        The sth element of which is a sparse matrix of size (nRows x nObs), where the (i, j)th 
        element equals 1 if the alternative corresponding to the ith column in expVarsMan is 
        available to the decision-maker corresponding to the jth observation, and 0 otherwise.
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element of the returned matrix  is 1 if the alternative corresponding 
        to the ith row in the data file was chosen during observation j, and 0 otherwise.   
    obsAv : sprase matrix of size (nObs x nDms).
        The (i, j)th element of the returned matrix is 1 if observation i corresponds to 
        decision-maker j, and 0 otherwise.
    rowAv : sparse matrix of size (nRows x (nAlts * nDms)).
        The (i, ((n - 1) * nAlts) + j)th element of the returned matrix is 1 if the ith row 
        in the data file corresponds to the jth alternative and the nth decision-maker, 
        and 0 otherwise.  

    
    """ 
    # Class membership model
    nInds = inds.shape[0]

    # Class-specific model
    altAvTuple, altChosen, obsAv, rowAv = processClassSpecificPanel(inds, indID, obsID, altID, choice)
    nRows = altID.shape[0]
    nObs = np.unique(obsID).shape[0]
    print('VARIABLE', altAvTuple)

    altAv = []
    for s in range(0, nClasses):
        altAv.append(coo_matrix((imposeCSConstraints(altID, availAlts[s]), #this constrains in this alternative is not available for that class
                (altAvTuple[0], altAvTuple[1])), shape = (nRows, nObs)))
    

    return (nInds,
            altAv, altChosen, obsAv, rowAv) 

    
def enumClassSpecificProbPanel(param, expVars, altAvMat, obsAv, rowAv, nDms, nAlts):
    """
    Function that calculates and enumerates the class specific choice probabilities 
    for each decision-maker in the sample and for each of the available alternatives
    in the choice set.
    """

    v = np.dot(param[None, :], expVars)               # v is 1 x nRows
    ev = np.exp(v)                                    # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                          # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                          # As precaution when exp(v) is too close to zero
    nev = ev * altAvMat                               # nev is 1 x nObs
    nnev = altAvMat * np.transpose(nev)               # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))             # p is 1 x nRows
    p[np.isinf(p)] = 1e-200                           # When none of the alternatives are available
    pAlt = p * rowAv                                  # pAlt is 1 x (nAlts * nDms)
    return pAlt.reshape((nDms, nAlts), order = 'C')
    

def calProb(nClasses, nInds, #paramClassMem, expVarsClassMem, indClassAv,
        paramClassSpec, expVarsClassSpec, altAv, altChosen, obsAv, indWeights, Q_train, weights, N_EPOCHS, iterCounter,n_nodes, n_latent_vars,ind, Q_dummies, outputFileName, Q_ID, model):
    """
    Function that calculates the expectation of the latent variables in E-Step of the 
    EM Algorithm and the value of the log-likelihood function.
    
    Returns
    -------
    weights : 2D numpy array of size (nClasses x nDms).
        The expected value of latent variable for each individual and each of the available
        classes.
    ll : Float.
        The value of log-likelihood.
    """    
    
    #Probability of the class for each individual
    pIndClass, history, model = calClassMemProb(Q_train, nClasses, weights, N_EPOCHS, iterCounter, n_nodes,n_latent_vars,ind, Q_dummies, outputFileName, Q_ID, model)
    pIndClass = pIndClass.T
    p = calClassSpecificProbPanel(paramClassSpec[0], expVarsClassSpec[0], altAv[0], altChosen, obsAv)
    for s in range(1, nClasses):
        p = np.vstack((p, calClassSpecificProbPanel(paramClassSpec[s], expVarsClassSpec[s], altAv[s], altChosen, obsAv)))
    #print('Choice prob of the 1st individual in both classes of all the choice she/he made:', p[0,:], p[1,:])
    #multiply arguments element-wise
    weights = np.multiply(p, pIndClass)
    print('Weights after multiplication', weights.shape)
    ll = np.sum(np.multiply(np.log(np.sum(weights, axis = 0)), indWeights))
    weights = np.divide(weights, np.tile(np.sum(weights, axis = 0), (nClasses, 1)))     # nClasses x nInds
    return weights, ll, history, model

#Random initialization of the classes
def initClassMem(Q_train, nClasses):
    init=[]
    for i in range (Q_train.shape[0]*nClasses):
        k=np.array(1/nClasses)
        init.append(k)
    init=np.reshape(init, (Q_train.shape[0], nClasses))
    return init
    
def emAlgo(outputFilePath, outputFileName, outputFile, nClasses, 
        indID, Q_dummies, availIndClasses, 
        obsID, altID, indID_test, obsID_test, altID_test, choice, choice_test, nchoices, availAlts, IniClassMem, expVarsClassSpec, expVarsClassSpec_test, namesExpVarsClassSpec, indWeights, indWeights_test, paramClassMem, paramClassSpec, Q_train, N_EPOCHS, Q_vars, n_nodes, 
             n_latent_vars, ind, Q_test, data_test, Q_dummies_test, iter, Q_ID, Q_ID_test): 
    """
    Function that implements the EM Algorithm to estimate the desired model specification. 

    """ 
    
    ###Process the data and the model structure
    startTime = datetime.now()
    print('Processing data')
    outputFile.write('Processing data\n')

    inds = np.unique(indID)

    (nInds,
      altAv, altChosen, obsAv, rowAv) \
        = processData(inds, indID, 
        nClasses, availIndClasses, 
        obsID, altID,
         choice, availAlts) 

    #Create the NN model structure
    optimizer = Adam(clipnorm= 50.)
        
    dict_losses={}
    dict_losses['Class']='categorical_crossentropy'

    for i in range(ind.shape[-1]):
        dict_losses["Output_ord_act"+ str(i+1)]='sparse_categorical_crossentropy'

    #number of variables in the latent variable model
    num_vars_indicators = Q_dummies.shape[-1]
    #number of variables in the class membership model
    socio_num = Q_train.shape[-1]
    #number of decision makers
    n_ID = Q_ID.shape[-1]

    mnl_classes=Latent_Class_Model(nIDs= n_ID, indicators_vars= num_vars_indicators, socio_num= socio_num,
                                   classes_num=nClasses,n_nodes=n_nodes,n_latent_vars= n_latent_vars)

    mnl_classes.compile(optimizer= optimizer, metrics= ["accuracy"], loss= dict_losses)


    Ascs_Train= np.zeros((len(Q_train),nClasses-1, nClasses,1))
    for i in range(nClasses-1):
        Ascs_Train[:,i,i,:]=1
    
    if Q_test!= 'None':
        ### Process data for the test
        inds_test = np.unique(indID_test)
        availIndClasses_test = np.ones((nClasses, data_test.shape[0]), dtype=np.int)
        (nInds_test, altAv_test, altChosen_test, obsAv_test, rowAv_test) \
        = processData(inds_test, indID_test, 
        nClasses, availIndClasses_test, 
        obsID_test, altID_test, choice_test, availAlts)

        #create Asc for the test
        Ascs_Test= np.zeros((len(Q_test),nClasses-1, nClasses,1))
        for i in range(nClasses-1):
            Ascs_Test[:,i,i,:]=1


    #### Start the iterations of the model
    print('Initializing EM Algorithm...\n')
    outputFile.write('Initializing EM Algorithm...\n\n')
    converged, iterCounter, llOld = False, 0, 0
    weights= IniClassMem
    #Weights is the probability for each class and individual in format [number of classes, nindividuals]
    weights = weights.T
    indWeights=indWeights.T
    while not converged:

        for s in range(0, nClasses):
            #Probability weighted
            cWeights = np.multiply(weights[s, :], indWeights)
            paramClassSpec[s] = minimize(wtLogitPanel, paramClassSpec[s], args = (expVarsClassSpec[s], altAv[s], 
                    cWeights, altAv[s] * obsAv * cWeights[:, None], altChosen, 
                    obsAv, choice), method = 'BFGS', jac = True, tol = llTol, options = {'gtol': grTol})['x']
        
        # E-Step: Calculate the expectations of the latent variables, using the current values for the model parameters.
        weights = weights.T
        weights, llNew, history, mnl_classes = calProb(nClasses, nInds,
                paramClassSpec, expVarsClassSpec, altAv, altChosen, obsAv,indWeights, Q_train, weights, N_EPOCHS, iterCounter, n_nodes, n_latent_vars,ind, Q_dummies, outputFileName, Q_ID, mnl_classes)
        
        tolerance = abs(llNew - llOld)

        currentTime = datetime.now().strftime('%a, %d %b %Y %H:%M:%S')
        print(currentTime,' Iteration:',iterCounter, ' ', llNew)

        #### Calculate the validation likelihood

        #loading the saved model
        #mnl_classes.load_weights('Models/Model_classes'+outputFileName+ str(iterCounter+1)+'.h5')

        if Q_test!= 'None':
            #Get the class test probabilities
            predictions= mnl_classes.predict(x= {'ascs_for_classes': Ascs_Test, 'input_dummies':Q_dummies_test, 'input_sc': Q_test, 'input_ID': Q_ID_test})
            p=predictions[0].T
            p1 = calClassSpecificProbPanel(paramClassSpec[0], expVarsClassSpec_test[0], altAv_test[0], altChosen_test, obsAv_test)
            for s in range(1, nClasses):
                p1 = np.vstack((p1, calClassSpecificProbPanel(paramClassSpec[s], expVarsClassSpec_test[s], altAv_test[s], altChosen_test, obsAv_test)))   
            prob = np.multiply(p, p1)
            lltest = np.sum(np.log(np.sum(prob, axis = 0)))
            print('Test loglikelihood:', lltest)

        outputFile = open(outputFilePath + outputFileName + 'Log.txt', 'a')
        outputFile.write('<%s> Iteration %d: %.4f  Tolerance %.4f Validation likelihood: %.4f\n' %(currentTime, (iterCounter+1), llNew, tolerance, lltest)) 
        outputFile.close()

        if (tolerance<emTol):
            converged = 1
        llOld = llNew
        iterCounter += 1
    
    outputFile = open(outputFilePath + outputFileName + 'Log.txt', 'a')
    outputFile.write('Converged :)') 
    outputFile.close()
    
    #### calculating the null log-likelihod: null parameters in both models
    
    paramClassSpecNull = []    
    for s in range(0, nClasses):
        paramClassSpecNull.append(np.zeros(expVarsClassSpec[s].shape[0]))
    initweights= initClassMem(Q_train, nClasses)
    #weightsNull, llNull, historyNull =  calProb(nClasses, nInds,
    #    paramClassSpecNull, expVarsClassSpec, altAv, altChosen, obsAv, indWeights, Q_train, initweights, N_EPOCHS, -1, n_nodes, n_latent_vars,ind, Q_dummies, outputFileName, Q_ID)
    p= initweights.T
    p1 = calClassSpecificProbPanel(paramClassSpecNull[0], expVarsClassSpec[0], altAv[0], altChosen, obsAv)
    for s in range(1, nClasses):
         p1 = np.vstack((p1, calClassSpecificProbPanel(paramClassSpecNull[s], expVarsClassSpec[s], altAv[s], altChosen, obsAv)))   
    prob = np.multiply(p, p1)
    llNull = np.sum(np.log(np.sum(prob, axis = 0)))
    
    ###### Getting the sample enumeration and save it

    print ('\nEnumerating choices for the sample')
    
    nAlts = np.unique(altID).shape[0]

    # performing sample enumeration to output the required csv file (ModelResultsSampleEnum.csv)
    predictions= mnl_classes.predict(x= {'input_sc': Q_train, 'ascs_for_classes': Ascs_Train, 'input_dummies':Q_dummies, 'input_ID': Q_ID})
    pIndMod=predictions[0]

    p = enumClassSpecificProbPanel(paramClassSpec[0], expVarsClassSpec[0], altAv[0], obsAv, rowAv, nInds, nAlts)
    for s in range(1, nClasses):
        p = np.hstack((p, enumClassSpecificProbPanel(paramClassSpec[s], expVarsClassSpec[s], altAv[s], obsAv, rowAv, nInds, nAlts)))
    p = np.hstack((inds[:, None], pIndMod, p))
    np.savetxt('SampleEnum/'+ outputFileName + 'SampleEnum.csv', p, delimiter = ',') 


    #### Get Rs and save them

    layer_name='Output_new_feature'
    inp = mnl_classes.input                                         
    layer =  mnl_classes.get_layer(layer_name)
    output= layer.output #mnl_classes.layers

    func = K.function([inp, K.learning_phase()], [output])
    model_inputs= [Ascs_Train, Q_dummies]
    layer_outs = func([model_inputs, 0])

    with open('Rs/Rs'+ outputFileName +'.pickle', 'wb') as handle:
        pickle.dump(layer_outs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #### Make a plot of the latent variables and the probabilities for each class
    if n_latent_vars==2:
        fig, ax = plt.subplots(nClasses, 1, figsize=(10, 20))
        for i in range(nClasses):
            class_probs=predictions[0].T[i]
            ax[i].scatter(layer_outs[0].T[0],layer_outs[0].T[1], c=class_probs, cmap=cm.coolwarm)
            ax[i].set_xlabel('R1')
            ax[i].set_ylabel('R2')
        fig.suptitle("Latent variables and probability of classes")
        fig.savefig(outputFileName+'Probabilities'+'.png')

    if n_latent_vars==3:
        fig, ax = plt.subplots(nClasses, n_latent_vars, figsize=(20, 14))
        for i in range(nClasses):
            class_probs=predictions[0].T[i]
            ax[i,0].scatter(layer_outs[0].T[0],layer_outs[0].T[1], c=class_probs, cmap=cm.coolwarm)
            ax[i,0].set_xlabel('R1')
            ax[i,0].set_ylabel('R2')
            ax[i,1].scatter(layer_outs[0].T[0],layer_outs[0].T[2], c=class_probs, cmap=cm.coolwarm)
            ax[i,1].set_xlabel('R1')
            ax[i,1].set_ylabel('R3')
            ax[i,2].scatter(layer_outs[0].T[1],layer_outs[0].T[2], c=class_probs, cmap=cm.coolwarm)
            ax[i,2].set_xlabel('R2')
            ax[i,2].set_ylabel('R3')
        fig.suptitle("Latent variables and probability of classes")
        fig.savefig(outputFileName+'Probabilities'+'.png')


    #Calculate standard errors of class especific
    #weights = weights.T
    stdErrClassSpec = []
    for s in range(0, nClasses):
        stdErrClassSpec.append(calStdErrWtLogitPanel(paramClassSpec[s], expVarsClassSpec[s], altAv[s], 
                    weights[s, :], altAv[s] * obsAv * weights[s, :][:, None], 
                    altChosen, obsAv, choice))
    
    ##### Display model fit results and parameter estimation results
               
    num_param_total= displayOutput(outputFile, startTime, llNew, llNull,
                  nClasses, Q_vars, nchoices,
            namesExpVarsClassSpec, paramClassSpec, stdErrClassSpec,obsID, n_latent_vars, n_nodes, outputFilePath, outputFileName, iterCounter, nInds, num_vars_indicators)
    
    weights = weights.T
    outputList= [weights]
    
    for i in ind.T:
        outputList.append(np.array(i).T) 
        
    stats_EMNL, dfinds = model_summary(mnl_classes, [Ascs_Train, Q_dummies, Q_train, Q_ID], outputList,  ind, n_latent_vars, outputFileName, Q_vars_names= ['ASC_Class_'+str(i+1) for i in range(nClasses-1)]+['Beta '+ Q_vars[j]+ ' class_' +str(x+1) for j in range(len(Q_vars)) for x in range((nClasses-1))]+['R'+str(i+1) + ' class_' + str(j+1) for i in range(n_latent_vars) for j in range(nClasses-1)]+['w_class'+ str(j+1) for j in range(nClasses-1)])
    print(stats_EMNL)
    print('\n\nUtitilities for the indicators:')
    print(dfinds)  

    #### Write results into the file

    f= open(outputFilePath + outputFileName + 'Param.txt','a')
    f.write('\nClass Membership Model:'+'\n')
    f.write(stats_EMNL.to_string())  
    f.write('\n\nUtilities for the indicators:'+'\n')
    f.write(dfinds.to_string())  
    f.write('\n \n Loss: '+ str(history['loss'][-1]))
    for i in range(ind.shape[-1]):
        f.write('\n Indicator '+str(i+1)+' accuracy: '+str(history['Output_ord_act' + str(i+1) + '_accuracy'][-1])) 
    f.close()
    f.close()

    #### TEST PART

    if Q_test == 'None':
        f= open(outputFilePath + outputFileName + 'Param.txt','a') 
        f.write('\nNo test'+'\n')
        f.close()

    else:

        #Calculate null test likelihood
        paramClassSpecNull = []    
        for s in range(0, nClasses):
            paramClassSpecNull.append(np.zeros(expVarsClassSpec_test[s].shape[0]))
        initweights= initClassMem(Q_test, nClasses)
        p= initweights.T
        p1 = calClassSpecificProbPanel(paramClassSpecNull[0], expVarsClassSpec_test[0], altAv_test[0], altChosen_test, obsAv_test)
        for s in range(1, nClasses):
            p1 = np.vstack((p1, calClassSpecificProbPanel(paramClassSpecNull[s], expVarsClassSpec_test[s], altAv_test[s], altChosen_test, obsAv_test)))   
        prob = np.multiply(p, p1)
        lltestnull = np.sum(np.log(np.sum(prob, axis = 0)))
        print('Test null loglikelihood:', lltestnull)

        #Get the class test probabilities
        predictions= mnl_classes.predict(x= {'ascs_for_classes': Ascs_Test, 'input_dummies':Q_dummies_test, 'input_sc': Q_test, 'input_ID': Q_ID_test})
        pfile=predictions[0]
        p=predictions[0].T
        #p= np.array(pfile).reshape(1,Q_test.shape[0],nClasses).reshape((nClasses, nInds_test), order = 'F')
        p1 = calClassSpecificProbPanel(paramClassSpec[0], expVarsClassSpec_test[0], altAv_test[0], altChosen_test, obsAv_test)
        for s in range(1, nClasses):
            p1 = np.vstack((p1, calClassSpecificProbPanel(paramClassSpec[s], expVarsClassSpec_test[s], altAv_test[s], altChosen_test, obsAv_test)))   
        prob = np.multiply(p, p1)
        lltest = np.sum(np.log(np.sum(prob, axis = 0)))
        print('Test loglikelihood:', lltest)
        #save the class probabilities in a file
        pfile = np.hstack((inds_test[:, None], pfile))
        np.savetxt('SampleEnum/'+ outputFileName + 'SampleEnumTest.csv', pfile, delimiter = ',') 
        rho_squaredtest = 1 - lltest/lltestnull
        rho_bar_squaredtest = 1 - (lltest-num_param_total)/lltestnull

        #Write into file
        f= open(outputFilePath + outputFileName + 'Param.txt','a') 
        f.write("\n\nNumber of Observations in the test:".ljust(45, ' ')+(str(np.unique(obsID_test).shape[0]).rjust(10,' '))+'\n')
        f.write("Number of individuals in the test:".ljust(45, ' ')+(str(nInds_test)).rjust(10,' ')+'\n') 
        f.write("\nNull test Log-Likelihood:".ljust(45, ' ')+(str(round(lltestnull,2)).rjust(10,' '))+'\n')
        f.write(("Fitted test Log-Likelihood:".ljust(45, ' ')+(str(round(lltest,2)).rjust(10,' ')))+'\n')
        f.write("Test Rho-Squared:".ljust(45, ' ')+(str(round(rho_squaredtest,2)).rjust(10,' '))+'\n')
        f.write("Test Rho-Bar-Squared:".ljust(45, ' ')+(str(round(rho_bar_squaredtest,2)).rjust(10,' '))+'\n')
        f.close()

    return mnl_classes, layer_outs


def lccm_fit(data,
             Q_train,
             Q_vars,
             Q_ID,
             Q_ID_test,
             n_nodes, 
             n_latent_vars,
             ind,
             Q_dummies,
             N_EPOCHS,
             ind_id_col, 
             obs_id_col,
             alt_id_col,
             choice_col,
             n_classes,
             class_specific_specs,
             class_specific_labels,
             iter,
             Q_test= np.array([]),
             IniClassMem = None,
             data_test= None,
             Q_dummies_test=None,
             indWeights = None,
             indWeights_test = None,
             avail_classes = None,
             avail_alts = None,
             paramClassMem = None,
             paramClassSpec = None,
             outputFilePath = '', 
             outputFileName = 'ModelResults'):

    outputFile = open(outputFilePath + outputFileName + 'Log.txt', 'w')
    
    # Generate columns representing individual, observation, and alternative id
    indID = data[ind_id_col].values
    obsID = data[obs_id_col].values
    altID = data[alt_id_col].values

    if Q_test == 'None':
        indID_test = None
        obsID_test = None
        altID_test = None
        choice_test = None

    else:
        indID_test = data_test[ind_id_col].values
        obsID_test = data_test[obs_id_col].values
        altID_test = data_test[alt_id_col].values
        choice_test = np.reshape(data_test[choice_col].values, (data_test.shape[0], 1))

    
    # Generate the choice column and add one dimension
    choice = np.reshape(data[choice_col].values, (data.shape[0], 1))
    #number of different alternatives
    nchoices= len(np.unique(altID))
    
    # Number of classes
    nClasses = n_classes
    
    # AVAILABLE CLASSES: Which latent classes are available to which decision-maker? 
    # 2D array of size (nClasses x nRows) where 1=available i.e. latent class is 
    #available to thee decision-maker in that row of that data and 0 otherwise
    
    #In this code it is assume that all the classes are available to all decision-makers
    if avail_classes is None:
        availIndClasses = np.ones((nClasses, data.shape[0]), dtype=np.int)

    #Random initialization of the class membership is none provided
    if IniClassMem is None:
        IniClassMem = initClassMem(Q_train, nClasses)
    

    # AVAILABLE ALTERNATIVES: Which choice alternatives are available to each latent
    # class of decision-makers? List of size nClasses, where each element is a list of
    # identifiers of the alternatives available to class membership

    # Default case is to make all alternative available to all latent classes
    if avail_alts is None:
        availAlts = [np.unique(altID) for s in class_specific_specs]  
    else:
        availAlts = avail_alts
    
    # CLASS-SPECIFIC MODELS: Use PyLogit to generate design matrices of explanatory variables
    # for each of the class specific choice models, including an intercept as specified by the user.
    
    design_matrices = [pylogit.choice_tools.create_design_matrix(data, spec, alt_id_col)[0] 
    						for spec in class_specific_specs]

    expVarsClassSpec = [np.transpose(m) for m in design_matrices]
    
    if Q_test== 'None':
        design_matrices_test = None
        expVarsClassSpec_test = None

    else:
        design_matrices_test = [pylogit.choice_tools.create_design_matrix(data_test, spec, alt_id_col)[0] 
    						for spec in class_specific_specs]

        expVarsClassSpec_test = [np.transpose(m) for m in design_matrices_test]

        if indWeights_test is None:    
            indWeights_test = np.ones((np.unique(indID_test).shape[0]))
    
    # NOTE: class-specific choice specifications with explanatory variables that vary
    # by alternative should work automatically thanks to PyLogit, but the output labels 
    # WILL NOT work until we update the LCCM code to handle that. 
    

    # Starting values for the parameters of the class specific models
    # making the starting value of the class membership and class specfic choice models random
    if paramClassSpec is None:
        paramClassSpec = []
        for s in range(0, nClasses):
            paramClassSpec.append(-np.random.rand(expVarsClassSpec[s].shape[0])/10)
    
    # Weights to account for choice-based sampling
    # By default the weights will be assumed to be equal to one for all individual
    # indWeights is 1D numpy array of size nInds accounting for the weight for each individual in the sample
    # as given by the user
    if indWeights is None:    
        indWeights = np.ones((np.unique(indID).shape[0]))

    
    # defining the names of the explanatory variables for class specific model
    # getting the requried list elements that comprise string of names of
    # explanatory variables to be used in displaying parameter estimates in the output tables.
    namesExpVarsClassSpec = []
    for i in range(0, len(class_specific_labels)):
        name_iterator=[]
        for key, value in class_specific_labels[i].items() :
            if type(value) is list:
                name_iterator += value
            else:
                name_iterator.append(value)
        namesExpVarsClassSpec.append(name_iterator)

    # Invoke emAlgo()
    model, layer_outs = emAlgo(outputFilePath = outputFilePath, 
           outputFileName = outputFileName, 
           outputFile = outputFile, 
           nClasses = nClasses, 
           indID = indID,
           Q_dummies = Q_dummies,
           availIndClasses = availIndClasses,
           obsID = obsID, 
           altID = altID,
           indID_test = indID_test,
           obsID_test = obsID_test,
           altID_test = altID_test,
           choice = choice,
           choice_test = choice_test,
           nchoices=nchoices,
           availAlts = availAlts,
           IniClassMem = IniClassMem,
           expVarsClassSpec = expVarsClassSpec,
           expVarsClassSpec_test = expVarsClassSpec_test,
           namesExpVarsClassSpec = namesExpVarsClassSpec, 
           indWeights = indWeights,
           indWeights_test = indWeights_test,
           paramClassMem = paramClassMem,
           paramClassSpec = paramClassSpec,
           Q_train= Q_train,
           N_EPOCHS=N_EPOCHS, Q_vars=Q_vars, n_nodes=n_nodes, 
           n_latent_vars=n_latent_vars,
           ind=ind,
           Q_test=Q_test,
           data_test=data_test,
           Q_dummies_test=Q_dummies_test,
           iter= iter,
           Q_ID= Q_ID,
           Q_ID_test = Q_ID_test)
    
    outputFile.close()
    return model, layer_outs
    
    
    