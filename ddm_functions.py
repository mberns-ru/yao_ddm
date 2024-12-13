import os
import matplotlib.pyplot as plt
import pyddm as ddm
import numpy as np
import pandas as pd
from pyddm import Model, Fittable, plot, InitialCondition, ICPoint
from pyddm.functions import fit_adjust_model, display_model, fit_model
from pyddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture, LossRobustBIC, LossRobustLikelihood
from pyddm.functions import fit_adjust_model
from ddm_models_2 import *
from data_parsing import *


#########################################
# COMPREHENSIVE FUNCTIONS
#########################################

def run_ddm_on_csv(file_path, title, proj_path, model_type='basic', params_csv = None, sim_params_csv = None):
    """
    Run drift diffusion modeling on a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing the data.
    title (str): The title of the model (e.g. "F13 Pre-HL Aud").
    proj_path (str): The path to the project folder.

    Returns:
    Model: The fitted model.
    """

    # Make appropriate references
    if not os.path.exists(proj_path + "\\graphs"):
        os.makedirs(proj_path + "\\graphs")
    graphs_folder = proj_path + "\\graphs"

    if params_csv is None:
        if not os.path.exists(proj_path + "\\model fits"):
            os.makedirs(proj_path + "\\model fits")
        params_csv = proj_path + "\\model fits\\" + model_type + " parameters.csv"

    if sim_params_csv is None:
        if not os.path.exists(proj_path + "\\model fits"):
            os.makedirs(proj_path + "\\model fits")
        sim_params_csv = proj_path + "\\model fits\\" + model_type + " simulated parameters.csv"

    if model_type == 'basic':
         
        # Create samples from the CSV file
        av_samp, a_samp, v_samp = create_basic_samples_from_csv(file_path)

        # Fit basic model
        if av_samp is not None:
            run_basic(av_samp, title + " AV", 'av', proj_path, graphs_folder, params_csv, sim_params_csv)
            #return av_samp
        if a_samp is not None:
            run_basic(a_samp, title + " Aud.", 'a', proj_path, graphs_folder, params_csv, sim_params_csv)
            #return a_samp
        if v_samp is not None:
            run_basic(v_samp, title + " Vis.", 'v', proj_path, graphs_folder, params_csv, sim_params_csv)
            #return v_samp

    elif model_type == 'bias':

        # Create samples from the CSV file
        av_samp, a_samp, v_samp = create_bias_samples_from_csv(file_path)

        # Fit bias model
        if av_samp is not None:
            run_bias(av_samp, title + " AV", 'av', proj_path, graphs_folder, params_csv, sim_params_csv)
            #return av_samp
        if a_samp is not None:
            run_bias(a_samp, title + " Aud.", 'a', proj_path, graphs_folder, params_csv, sim_params_csv)
            #return a_samp
        if v_samp is not None:
            run_bias(v_samp, title + " Vis.", 'v', proj_path, graphs_folder, params_csv, sim_params_csv)
            #return v_samp

    return

def run_basic(samp, title, trial_type, proj_path, graphs_folder, params_csv, sim_params_csv):

    print("Fitting model...")
    fitted_og_model = fit_basic(samp, title, trial_type, graphs_folder, params_csv)

    print("Simulating model...")
    fitted_og_model, fitted_sim_model, samp, new_samp = simulate_model(fitted_og_model, samp, title, trial_type, graphs_folder, sim_params_csv)

    print("Validating simulation...")
    try:
        validate_simulated_fit(params_csv, sim_params_csv, graphs_folder)
    except:
        print('Error when validating.')

    print("Graphing trials...")
    graph_basic_trial(samp, title, graphs_folder)

    print("Done! All graphs and parameters saved.")

    return

def run_bias(samp, title, trial_type, proj_path, graphs_folder, params_csv, sim_params_csv):

    print("Fitting model...")
    fitted_og_model = fit_bias(samp, title, trial_type, graphs_folder, params_csv)

    print("Simulating model...")
    fitted_og_model, fitted_sim_model, samp, new_samp = simulate_bias_model(fitted_og_model, samp, title, trial_type, graphs_folder, sim_params_csv)

    print("Validating simulation...")
    validate_simulated_bias_fit(params_csv, sim_params_csv, graphs_folder)

    print("Graphing trials...")
    graph_bias_trial(samp, title, graphs_folder)

    print("Done! All graphs and parameters saved.")

    return

#########################################
# DATA PARSING
#########################################

def create_basic_samples_from_csv(file_path):
    """
    Create samples for drift diffusion modeling from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing the data.

    Returns:
    tuple: A tuple containing three samples: av_samp, a_samp, v_samp.
        - av_samp: Sample for audio-visual trials.
        - a_samp: Sample for auditory trials.
        - v_samp: Sample for visual trials.

        Samples may be none.
    """
    df_rt = pd.read_csv(file_path)

    # Convert latency to seconds
    df_rt['Latency'] /= 1000

    # Remove short and long trials
    # df_rt = df_rt[df_rt["Latency"] > .1] # Remove trials less than 100ms
    df_rt = df_rt[df_rt["Latency"] < 2.5] # Remove trials greater than 1650ms

    # Remove catch trials
    mask = (df_rt['Score'] == 'Correct') | (df_rt['Score'] == 'Incorrect')
    df_rt = df_rt[mask]

    # Change correct and incorrect to 1 and 0
    df_rt = df_rt.replace('Correct', 1)
    df_rt = df_rt.replace('Incorrect', 0)

    # Change values into correct values
    df_rt['AMRate'].replace(10, 9.77, inplace=True)
    df_rt['AMRate'].replace(8, 7.82, inplace=True)

    # Separate into trial types
    df_rt_av = df_rt[df_rt["TrialType"] == 'Av']
    if len(df_rt_av) == 0:
        av_samp = None
    else:
        av_samp = ddm.Sample.from_pandas_dataframe(df_rt_av, rt_column_name="Latency", choice_column_name="Score")

    df_rt_a = df_rt[df_rt["TrialType"] == 'Aud']
    # print(df_rt_a)
    if len(df_rt_a) == 0:
        a_samp = None
    else:
        a_samp = ddm.Sample.from_pandas_dataframe(df_rt_a, rt_column_name="Latency", choice_column_name="Score")

    df_rt_v = df_rt[df_rt["TrialType"] == 'Vis']
    if len(df_rt_v) == 0:
        v_samp = None
    else:
        v_samp = ddm.Sample.from_pandas_dataframe(df_rt_v, rt_column_name="Latency", choice_column_name="Score")

    # Return all samples, even if None
    return av_samp, a_samp, v_samp

def create_bias_samples_from_csv(file_path):
    """
    Create samples for a biased drift diffusion model using stimulus coding from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing the data.

    Returns:
    tuple: A tuple containing three samples: av_samp, a_samp, v_samp.
        - av_samp: Sample for audio-visual trials.
        - a_samp: Sample for auditory trials.
        - v_samp: Sample for visual trials.

        Samples may be none.
    """
    df_rt = pd.read_csv(file_path)

    # Convert latency to seconds
    df_rt['Latency'] /= 1000

    # Remove short and long trials
    # df_rt = df_rt[df_rt["Latency"] > .1] # Remove trials less than 100ms
    df_rt = df_rt[df_rt["Latency"] < 2.5] # Remove trials greater than 1650ms

    # Remove catch trials
    mask = (df_rt['Score'] == 'Correct') | (df_rt['Score'] == 'Incorrect')
    df_rt = df_rt[mask]

    # Change correct and incorrect to 1 and 0
    df_rt = df_rt.replace('Correct', 1)
    df_rt = df_rt.replace('Incorrect', 0)

    # Change values into correct values
    df_rt['AMRate'].replace(10, 9.77, inplace=True)
    df_rt['AMRate'].replace(8, 7.82, inplace=True)

    # Add "left_is_correct" column:
    left_corr_vals = (df_rt['AMRate'] < 6)
    left_corr_vals.replace(True,1, inplace=True)
    left_corr_vals.replace(False,0, inplace=True)
    df_rt.insert(column='left_is_correct', value=left_corr_vals, loc=len(df_rt.columns))

    # Add "choice_side" column:
    # 1 = left
    # 0 = right
    choice_side_vals = ((df_rt['AMRate'] < 6) & (df_rt['Score'] == 1)) | ((df_rt['AMRate'] > 6) & (df_rt['Score'] == 0))
    choice_side_vals.replace(True,1, inplace=True)
    choice_side_vals.replace(False,0, inplace=True)
    df_rt.insert(column='choice_side', value=choice_side_vals, loc=len(df_rt.columns))

    # Separate into trial types
    df_rt_av = df_rt[df_rt["TrialType"] == 'Av']
    if len(df_rt_av) == 0:
        av_samp = None
    else:
        av_samp = ddm.Sample.from_pandas_dataframe(df_rt_av, rt_column_name="Latency", choice_column_name="choice_side", choice_names=("Left", "Right"))

    df_rt_a = df_rt[df_rt["TrialType"] == 'Aud']
    if len(df_rt_a) == 0:
        a_samp = None
    else:
        a_samp = ddm.Sample.from_pandas_dataframe(df_rt_a, rt_column_name="Latency", choice_column_name="choice_side", choice_names=("Left", "Right"))

    df_rt_v = df_rt[df_rt["TrialType"] == 'Vis']
    if len(df_rt_v) == 0:
        v_samp = None
    else:
        v_samp = ddm.Sample.from_pandas_dataframe(df_rt_v, rt_column_name="Latency", choice_column_name="choice_side", choice_names=("Left", "Right"))

    # Return all samples, even if None
    return av_samp, a_samp, v_samp

def create_basic_df_from_model(fitted_model):
    """
    Create a pandas DataFrame from a basic fitted model.

    Parameters:
    fitted_model (object): The fitted model object.

    Returns:
    df (DataFrame): The DataFrame containing the model parameters, animal ID, trial type, accuracy, and mean.
    """
    param_names = fitted_model.get_model_parameter_names()
    id_cols = ['animal_id', 'trial_type']
    all_cols = np.append(id_cols, param_names)
    all_cols = np.append(all_cols, ['accuracy', 'mean'])
    df = pd.DataFrame(columns=all_cols)
    return df

def create_bias_df_from_model(fitted_model):
    """
    Create a pandas DataFrame from a biased fitted model.

    Parameters:
    fitted_model (object): The fitted model object.

    Returns:
    df (DataFrame): The DataFrame containing the model parameters, animal ID, trial type, pRight and mean.
    """
    param_names = fitted_model.get_model_parameter_names()
    id_cols = ['animal_id', 'trial_type']
    all_cols = np.append(id_cols, param_names)
    all_cols = np.append(all_cols, ['pRight', 'mean'])
    df = pd.DataFrame(columns=all_cols)
    return df

def extract_basic_model_params(model_fit, df, name, trial_type, samp, csv_path):
    """
    Extracts model parameters from a basic model fit object and appends them to a DataFrame. Saves as a csv.

    Parameters:
        model_fit (object): The model fit object containing the model parameters.
        df (DataFrame): The DataFrame to which the extracted parameters will be appended.
        name (str): The name of the participant.
        trial_type (str): The type of trial.
        samp (object): The object containing the sample data.

    Returns:
        DataFrame: The updated DataFrame with the extracted parameters.

    """
    id_cols = [name, trial_type]
    param_vals = model_fit.get_model_parameters()
    new_row = np.append(id_cols, param_vals)
    samp_acc = samp.prob('correct')
    samp_mean = samp.mean_decision_time()
    new_row = np.append(new_row, [samp_acc, samp_mean])

    # Edge case: this data has already been run and is already in the csv
    mask = (df['animal_id'] == name) & (df['trial_type'] == trial_type)

    if mask.any():
        #print("Edge case detected.")
        df.loc[mask] = new_row
    else:
        df.loc[len(df)] = new_row

    # Saves parameters
    df.to_csv(csv_path, index=False)

    return df

def extract_bias_model_params(model_fit, df, name, trial_type, samp, csv_path):
    """
    Extracts model parameters from a model fit object and appends them to a DataFrame. Saves as a csv.

    Parameters:
        model_fit (object): The model fit object containing the model parameters.
        df (DataFrame): The DataFrame to which the extracted parameters will be appended.
        name (str): The name of the participant.
        trial_type (str): The type of trial.
        samp (object): The object containing the sample data.

    Returns:
        DataFrame: The updated DataFrame with the extracted parameters.

    """
    id_cols = [name, trial_type]
    param_vals = model_fit.get_model_parameters()
    new_row = np.append(id_cols, param_vals)
    samp_pright = samp.prob('Right')
    samp_mean = 0
    new_row = np.append(new_row, [samp_pright, samp_mean])

    # Edge case: this data has already been run and is already in the csv
    mask = (df['animal_id'] == name) & (df['trial_type'] == trial_type)

    if mask.any():
        #print("Edge case detected.")
        df.loc[mask] = new_row
    else:
        df.loc[len(df)] = new_row

    # Saves parameters
    df.to_csv(csv_path, index=False)

    return df

def fit_basic(samp, title, trial_type, graphs_folder, csv_path):
    """
    Fit a basic drift diffusion model to a sample. Saves to csv.

    Parameters:
    sample (Sample): The sample to fit the model to.
    title (str): The title of the model (e.g. "F13 Pre-HL Aud").

    Returns:
    Model: The fitted model.
    """
    model_fitted = fit_model(sample=samp,drift=DriftCoherence(drifttone=Fittable(minval=0, maxval=20)),
                 noise=NoiseConstant(noise=1),
                 bound=BoundConstant(B=Fittable(minval=.1, maxval=2)),
                 overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=0.8)),
                                                OverlayPoissonMixture(pmixturecoef=.02,
                                                                      rate=1)]),
                 dx=.001, dt=0.001, verbose=False)
    ddm.plot.plot_fit_diagnostics(model=model_fitted, sample=samp)
    # print("Parameters:", model_fitted.parameters())
    plt.title(title + ' Basic Fit ')
    plt.savefig(graphs_folder + "\\" + title + " basic fit.pdf", transparent=True,format="pdf")
    plt.close()

    if not os.path.exists(csv_path):
        #print("File doesn't exist.")
        fitted_df = create_basic_df_from_model(model_fitted)
    else:
        #print("File exists")
        fitted_df = pd.read_csv(csv_path)
    extract_basic_model_params(model_fitted, fitted_df, title, trial_type, samp, csv_path)

    return model_fitted

def fit_bias(samp, title, trial_type, graphs_folder, csv_path):
    """
    Fit a biased drift diffusion model to a sample. Saves to csv.

    Parameters:
    sample (Sample): The sample to fit the model to.
    title (str): The title of the model (e.g. "F13 Pre-HL Aud").

    Returns:
    Model: The fitted model.
    """
    model_fitted = Model(drift=DriftCoherence(drifttone=Fittable(minval=-20, maxval=20)),
                             noise=NoiseConstant(noise=1),
                             bound=BoundConstant(B=Fittable(minval=0.1, maxval=10)),
                             IC=ICPoint(x0=Fittable(minval=-0.1, maxval=0.1)),
                             overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=0.8)),
                                                            OverlayPoissonMixture(pmixturecoef=.02, rate=1)]),
                             T_dur=2.5, dx=.001, dt=0.001, choice_names = ('Left', 'Right'))
    model_fitted = fit_adjust_model(samp, model_fitted, verbose=False)
    #ddm.plot.model_gui(model=model_fitted, sample=samp)
    ddm.plot.plot_fit_diagnostics(model=model_fitted, sample=samp)
    plt.title(title + ' Biased Fit ')
    plt.savefig(graphs_folder + "\\" + title + " bias fit.pdf", transparent=True,format="pdf")
    plt.close()

    if not os.path.exists(csv_path):
        #print("File doesn't exist.")
        fitted_df = create_bias_df_from_model(model_fitted)
    else:
        #print("File exists")
        fitted_df = pd.read_csv(csv_path)
    extract_bias_model_params(model_fitted, fitted_df, title, trial_type, samp, csv_path)

    return model_fitted

#########################################
# SIMULATION AND VALIDATION OF SIMULATION
#########################################

def simulate_model(og_fit, samp, title, trial_type, graphs_folder, sim_params_csv, model_to_use=model_basic):
    """
    Simulates a model fit using artificial data and returns the model fits. Exports parameters of original and simulated models.

    Parameters:
    - og_fit: The original model fit to be simulated.
    - samp: The sample object containing the data.
    - title: The title of the model (e.g. "F13 Pre-HL Aud").
    - model_to_use: The model to use for simulation (default: model_basic).

    Returns:
    - og_fit: The original model fit.
    - fit_sim: The simulated model fit.
    - samp: The original sample object.
    - new_samp: The simulated sample object.
    """
    og_params = og_fit.get_model_parameters()

    # Generate aritificial data from the model
    combos = samp.condition_combinations()
    samp_sims = []
    for combo in combos:
        num_per_comb = (len(samp.subset(**combo)))
        sol = og_fit.solve(conditions=combo)
        samp_sim = sol.resample(num_per_comb)
        samp_sims.append(samp_sim)
    new_samp = samp_sims[0]
    for i in range(1,len(samp_sims)):
        new_samp = new_samp + samp_sims[i]

    # Fit model to artificial data
    fit_sim = fit_model(sample=new_samp, drift=DriftCoherence(drifttone=Fittable(minval=0, maxval=20)),
                 noise=NoiseConstant(noise=1),
                 bound=BoundConstant(B=Fittable(minval=.1, maxval=2)),
                 overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=0.8)),
                                                OverlayPoissonMixture(pmixturecoef=.02,
                                                                      rate=1)]),
                 dx=.001, dt=0.001, verbose=False)
    ddm.plot.plot_fit_diagnostics(model=fit_sim, sample=new_samp)
    plt.title(title + ' Simulated Basic Fit')
    plt.savefig(graphs_folder + "\\" + title + " basic simulated fit.pdf", transparent=True,format="pdf")
    plt.close()

    # Export simulation parameters
    if not os.path.exists(sim_params_csv):
        fitted_df = create_basic_df_from_model(fit_sim)
    else:
        fitted_df = pd.read_csv(sim_params_csv)
    extract_basic_model_params(fit_sim, fitted_df, title, trial_type, new_samp, sim_params_csv)

    # Return model fits
    return og_fit, fit_sim, samp, new_samp

def validate_simulated_fit(og_file, sim_file, graphs_folder):
    """
    Compare the accuracy and mean values between the original and simulated data.

    Parameters:
    og_file (str): The file path to the csv with the original fit parameters.
    sim_file (str): The file path to the csv with the simulated fit parameters.
    graphs_folder (str): The folder to save the graphs.

    Returns:
    None
    """

    og_df = pd.read_csv(og_file)
    sim_df = pd.read_csv(sim_file)

    fig, axs = plt.subplots(2)
    axs[0].scatter(og_df['accuracy'], sim_df['accuracy'])
    axs[1].scatter(og_df['mean'], sim_df['mean'])
    axs[0].axline((0,0), slope=1, c='gray', linestyle = '--')
    axs[1].axline((0,0), slope=1, c='gray', linestyle = '--')
    axs[0].set_title('pRight')
    axs[0].set_xlabel('Original')
    axs[0].set_ylabel('Simulated')
    axs[1].set_title('Mean')
    axs[1].set_xlabel('Original')
    axs[1].set_ylabel('Simulated')

    plt.tight_layout()
    plt.close()
    fig.savefig(graphs_folder + "\\basic fit validation.pdf", transparent=True,format="pdf")
    return

def graph_basic_trial(samp, title, graphs_folder):
    """
    Plot and save a basic simulated trial graph for each AMRate condition in the given sample.

    Parameters:
    samp (Sample): The sample object containing the data.
    title (str): The title of the graph.
    graphs_folder (str): The folder path where the graph will be saved.

    Returns:
    None
    """
    # Fit model
    model_fitted = fit_model(sample=samp, drift=DriftCoherence(drifttone=Fittable(minval=-20, maxval=20)),
                    noise=NoiseConstant(noise=1),
                    bound=BoundConstant(B=Fittable(minval=.1, maxval=2)),
                    overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=-0.8, maxval=0.8))]),
                    dx=.001, dt=0.001, verbose=False, lossfunction=LossRobustLikelihood)
    
    # Graph 20 different trials with different seeds
    #print(samp.condition_values('AMRate'))
    seed_count = 0
    for i in range(20):
        curr_alpha = 1
        curr_color = 'blue'
        reset_alpha = len(samp.condition_values('AMRate'))//2
        counter = 0
        for amrate in samp.condition_values('AMRate'):
            trajectory = model_fitted.simulate_trial(conditions={'AMRate': amrate}, seed=seed_count)
            seed_count += 1
            if counter == reset_alpha:
                curr_alpha = 0
            if amrate > 6:
                curr_color = 'red'
                trajectory = -trajectory
                curr_alpha += 1/(reset_alpha + 1)
            else:
                curr_alpha -= 1/(reset_alpha + 1)

            counter += 1

            times = model_fitted.t_domain()
            times = times[:len(trajectory)]
            plt.plot(times, trajectory, label = str(round(amrate, 2)) + " kHz", color = curr_color, alpha=curr_alpha)

        upperbound = model_fitted.parameters()['bound']['B']
        plt.xlabel('Time (s)')
        plt.ylabel('Decision Variable')
        plt.title(title + " Simulated Trial per AM Rate: " + str(i))
        plt.axhline(upperbound, color = 'gray', linestyle = '--')
        plt.axhline(-upperbound, color = 'gray', linestyle = '--')
        plt.legend()
        plt.savefig(graphs_folder + "\\" + title + " basic trial " + str(i) + ".pdf", transparent=True,format="pdf")
        plt.close()
    return

def graph_basic_pre_post_trial(pre_samp, post_samp, title, graphs_folder):
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'black', 'cyan']

    # Fit models
    model_fitted_pre = fit_model(sample=pre_samp, drift=DriftCoherence(drifttone=Fittable(minval=-20, maxval=20)),
                    noise=NoiseConstant(noise=1),
                    bound=BoundConstant(B=Fittable(minval=.1, maxval=2)),
                    overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=-0.8, maxval=0.8))]),
                    dx=.001, dt=0.001, verbose=False, lossfunction=LossRobustLikelihood)
    model_fitted_post = fit_model(sample=post_samp, drift=DriftCoherence(drifttone=Fittable(minval=-20, maxval=20)),
                    noise=NoiseConstant(noise=1),
                    bound=BoundConstant(B=Fittable(minval=.1, maxval=2)),
                    overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=-0.8, maxval=0.8))]),
                    dx=.001, dt=0.001, verbose=False, lossfunction=LossRobustLikelihood)
    
    seed_count = 0
    for i in range(40):

        counter = 0
        for amrate in pre_samp.condition_values('AMRate'):
            # Graph pre
            trajectory = model_fitted_pre.simulate_trial(conditions={'AMRate': amrate}, seed=seed_count)
            if amrate > 6:
                trajectory = -trajectory
            times = model_fitted_pre.t_domain()
            times = times[:len(trajectory)]
            plt.plot(times, trajectory, label = str(round(amrate, 2)) + " kHz", color = colors[counter])
            counter += 1
        
        counter = 0
        for amrate in post_samp.condition_values('AMRate'):
            # Graph pre
            trajectory = model_fitted_post.simulate_trial(conditions={'AMRate': amrate}, seed=seed_count)
            seed_count += 1
            if amrate > 6:
                trajectory = -trajectory
            times = model_fitted_pre.t_domain()
            times = times[:len(trajectory)]
            plt.plot(times, trajectory, label = str(round(amrate, 2)) + " kHz", color = colors[counter], linestyle='--')
            counter += 1

        #upperbound = model_fitted_pre.parameters()['bound']['B']
        plt.xlabel('Time (s)')
        plt.ylabel('Decision Variable')
        plt.title(title + " Simulated Trial per AM Rate: " + str(i))
        #plt.axhline(upperbound, color = 'gray', linestyle = '--')
        #plt.axhline(-upperbound, color = 'gray', linestyle = '--')
        plt.legend()


        plt.savefig(graphs_folder + "\\" + title + " basic trial pre-post " + str(i) + ".pdf", transparent=True,format="pdf")
        plt.close()

    return

def graph_avg_basic_trial(samp, title, graphs_folder):
    """
    Plot and save a basic simulated trial graph for each AMRate condition in the given sample.

    Parameters:
    samp (Sample): The sample object containing the data.
    title (str): The title of the graph.
    graphs_folder (str): The folder path where the graph will be saved.

    Returns:
    None
    """
    # Fit model
    model_fitted = fit_model(sample=samp, drift=DriftCoherence(drifttone=Fittable(minval=-20, maxval=20)),
                    noise=NoiseConstant(noise=1),
                    bound=BoundConstant(B=Fittable(minval=.1, maxval=2)),
                    overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=-0.8, maxval=0.8))]),
                    dx=.001, dt=0.001, verbose=False, lossfunction=LossRobustLikelihood)
    
    # Simulate 100 trajectories per AMRate, find average, plot
    for amrate in samp.condition_values('AMRate'):
        seed_count = 0
        trajectories = []
        for i in range(1000):
            trajectory = model_fitted.simulate_trial(conditions={'AMRate': amrate}, seed=i)
            if amrate > 6:
                trajectory = -trajectory
            trajectories.append(trajectory)

        avg_traj_len = np.mean([len(traj) for traj in trajectories])
        #print(avg_traj_len)
        cropped_trajectories = []
        for traj in trajectories:
            if len(traj) > avg_traj_len:
                traj = traj[:int(avg_traj_len)]
            elif len(traj) < avg_traj_len:
                traj = np.pad(traj, (0, int(avg_traj_len - len(traj))), 'constant', constant_values=np.nan)
            cropped_trajectories.append(traj)
        avg_traj = np.nanmean(cropped_trajectories, axis=0)
        times = model_fitted.t_domain()[:int(avg_traj_len)]
        plt.plot(times, avg_traj, label = str(round(amrate, 2)) + " kHz")
        
    upperbound = model_fitted.parameters()['bound']['B']
    plt.xlabel('Time (s)')
    plt.ylabel('Decision Variable')
    plt.title(title + " Average Basic Trial per AM Rate")
    #plt.axhline(upperbound, color = 'gray', linestyle = '--')
    #plt.axhline(-upperbound, color = 'gray', linestyle = '--')
    plt.legend()
    plt.savefig(graphs_folder + "\\" + title + " avg basic trial.pdf", transparent=True,format="pdf")
    plt.close()
    return


def simulate_bias_model(og_fit, samp, title, trial_type, graphs_folder, sim_params_csv, model_to_use=model_bias):
    """
    Simulates a model fit using artificial data and returns the model fits. Exports parameters of original and simulated models.

    Parameters:
    - og_fit: The original model fit to be simulated.
    - samp: The sample object containing the data.
    - title: The title of the model (e.g. "F13 Pre-HL Aud").
    - model_to_use: The model to use for simulation (default: model_bias).

    Returns:
    - og_fit: The original model fit.
    - fit_sim: The simulated model fit.
    - samp: The original sample object.
    - new_samp: The simulated sample object.
    """
    og_params = og_fit.get_model_parameters()

    # Generate aritificial data from the model
    combos = samp.condition_combinations()
    samp_sims = []
    for combo in combos:
        num_per_comb = (len(samp.subset(**combo)))
        sol = og_fit.solve(conditions=combo)
        samp_sim = sol.resample(num_per_comb)
        samp_sims.append(samp_sim)
    new_samp = samp_sims[0]
    for i in range(1,len(samp_sims)):
        new_samp = new_samp + samp_sims[i]

    # Fit model to artificial data
    fit_sim = Model(drift=DriftCoherence(drifttone=Fittable(minval=-20, maxval=20)),
                            noise=NoiseConstant(noise=1),
                            bound=BoundConstant(B=Fittable(minval=0.1, maxval=10)),
                            IC=ICPoint(x0=Fittable(minval=-0.1, maxval=0.1)),
                            overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=0.8)),
                                                        OverlayPoissonMixture(pmixturecoef=.02, rate=1)]),
                            T_dur=2.5, dx=.001, dt=0.001, choice_names=('Left', 'Right'))
    fit_sim = fit_adjust_model(sample=samp, model=fit_sim, verbose=False, lossfunction=LossRobustLikelihood)
    ddm.plot.plot_fit_diagnostics(model=fit_sim, sample=new_samp)
    plt.title(title + ' Simulated Biased Fit')
    plt.savefig(graphs_folder + "\\" + title + " bias simulated fit.pdf", transparent=True,format="pdf")
    plt.close()

    # Export simulation parameters
    if not os.path.exists(sim_params_csv):
        fitted_df = create_bias_df_from_model(fit_sim)
    else:
        fitted_df = pd.read_csv(sim_params_csv)
    extract_bias_model_params(fit_sim, fitted_df, title, trial_type, new_samp, sim_params_csv)

    # Return model fits
    return og_fit, fit_sim, samp, new_samp

def validate_simulated_bias_fit(og_file, sim_file, graphs_folder):
    """
    Compare the accuracy and mean values between the original and simulated data.

    Parameters:
    og_file (str): The file path to the csv with the original fit parameters.
    sim_file (str): The file path to the csv with the simulated fit parameters.
    graphs_folder (str): The folder to save the graphs.

    Returns:
    None
    """

    og_df = pd.read_csv(og_file)
    sim_df = pd.read_csv(sim_file)

    fig, axs = plt.subplots(2)
    axs[0].axline((0,0), slope=1, c='gray', linestyle = '--')
    axs[1].axline((0,0), slope=1, c='gray', linestyle = '--')
    axs[0].scatter(og_df['pRight'], sim_df['pRight'])
    axs[1].scatter(og_df['mean'], sim_df['mean'])
    axs[0].set_title('pRight')
    axs[0].set_xlabel('Original')
    axs[0].set_ylabel('Simulated')
    axs[1].set_title('Mean')
    axs[1].set_xlabel('Original')
    axs[1].set_ylabel('Simulated')

    plt.tight_layout()
    plt.close()
    plt.savefig(graphs_folder + "\\bias fit validation.pdf", transparent=True,format="pdf")
    return

def graph_bias_trial(samp, title, graphs_folder):
    """
    Plot and save a biased simulated trial graph for each AMRate condition in the given sample.

    Parameters:
    samp (Sample): The sample object containing the data.
    title (str): The title of the graph.
    graphs_folder (str): The folder path where the graph will be saved.

    Returns:
    None
    """
    model_fitted = Model(drift=DriftCoherence(drifttone=Fittable(minval=-20, maxval=20)),
                             noise=NoiseConstant(noise=1),
                             bound=BoundConstant(B=Fittable(minval=0.1, maxval=10)),
                             IC=ICPoint(x0=Fittable(minval=-0.1, maxval=0.1)),
                             overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=0.8))]),
                             T_dur = 2.5, dx=0.001, dt=0.001, choice_names=('Left', 'Right'))
    model_fitted = fit_adjust_model(sample=samp, model=model_fitted, verbose=False, lossfunction=LossRobustLikelihood)
    # ddm.plot.model_gui(model=model_fitted, sample=samp)
    seed_count = 0
    for amrate in samp.condition_values('AMRate'):
        if amrate < 6:
            corr_choice = "Left"
        else:
            corr_choice = "Right"
        trajectory = model_fitted.simulate_trial(conditions={'AMRate': amrate}, seed=0)
        seed_count += 1
        times = model_fitted.t_domain()
        times = times[:len(trajectory)]
        plt.plot(times, trajectory, label = str(round(amrate, 2)) + " kHz")
    upperbound = model_fitted.parameters()['bound']['B']
    plt.xlabel('Time (s)')
    plt.ylabel('Decision Variable')
    plt.title(title + " Simulated Trial per AM Rate")
    plt.axhline(upperbound, color = 'gray', linestyle = '--')
    plt.axhline(-upperbound, color = 'gray', linestyle = '--')
    plt.legend()
    plt.savefig(graphs_folder + "\\" + title + " bias trial.pdf", transparent=True,format="pdf")
    plt.close()
    return

def graph_avg_bias_trial(samp, title, graphs_folder):
    """
    Plot and save a bias simulated trial graph for each AMRate condition in the given sample.

    Parameters:
    samp (Sample): The sample object containing the data.
    title (str): The title of the graph.
    graphs_folder (str): The folder path where the graph will be saved.

    Returns:
    None
    """
    # Fit model
    model_fitted = Model(drift=DriftCoherence(drifttone=Fittable(minval=-20, maxval=20)),
                             noise=NoiseConstant(noise=1),
                             bound=BoundConstant(B=Fittable(minval=0.1, maxval=10)),
                             IC=ICPoint(x0=Fittable(minval=-0.1, maxval=0.1)),
                             overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=0.8))]),
                             T_dur = 2.5, dx=0.001, dt=0.001, choice_names=('Left', 'Right'))
    model_fitted = fit_adjust_model(sample=samp, model=model_fitted, verbose=False, lossfunction=LossRobustLikelihood)
    
    # Simulate 100 trajectories per AMRate, find average, plot
    for amrate in samp.condition_values('AMRate'):
        trajectories = []
        for i in range(100):
            #print(amrate, i)
            trajectory = model_fitted.simulate_trial(conditions={'AMRate': amrate}, seed=i)
            trajectories.append(trajectory)

        avg_traj_len = np.mean([len(traj) for traj in trajectories])
        #print(avg_traj_len)
        cropped_trajectories = []
        for traj in trajectories:
            if len(traj) > avg_traj_len:
                traj = traj[:int(avg_traj_len)]
            elif len(traj) < avg_traj_len:
                traj = np.pad(traj, (0, int(avg_traj_len - len(traj))), 'constant', constant_values=np.nan)
            cropped_trajectories.append(traj)
        avg_traj = np.nanmean(cropped_trajectories, axis=0)
        times = model_fitted.t_domain()[:int(avg_traj_len)]
        plt.plot(times, avg_traj, label = str(round(amrate, 2)) + " kHz")
        
    upperbound = model_fitted.parameters()['bound']['B']
    plt.xlabel('Time (s)')
    plt.ylabel('Decision Variable')
    plt.title(title + " Average Bias Trial per AM Rate")
    #plt.axhline(upperbound, color = 'gray', linestyle = '--')
    #plt.axhline(-upperbound, color = 'gray', linestyle = '--')
    plt.legend()
    plt.savefig(graphs_folder + "\\" + title + " avg bias trial.pdf", transparent=True,format="pdf")
    plt.close()
    return

def test_func():
    import matplotlib.pyplot as plt
    from pyddm import Model
    from pyddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, LossRobustLikelihood

    class DriftCoherence(ddm.models.Drift):
        name = "Drift depends linearly on AMRate distance from midpoint"
        required_parameters = ["drifttone"]
        required_conditions = ["AMRate"]

        def get_drift(self, conditions, **kwargs):
            return self.drifttone * (np.abs(np.log(conditions['AMRate']) - np.log(6.25)))

    rates = [4, 5, 7.22, 9.77]
    choices = [1, 1, 0, 0]

    dummy_model = Model(drift=DriftCoherence(drifttone=4),
                        noise=NoiseConstant(noise=1),
                        bound=BoundConstant(B=1),
                        IC=ICPoint(x0=0.05),
                        overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=0.3)]),
                        T_dur = 2.5, dx=0.001, dt=0.001, choice_names=('Left', 'Right'))
    
    ddm.plot.model_gui(dummy_model, conditions={'AMRate': rates})
    
    for i in range(4):
        traj = dummy_model.simulate_trial(conditions={'AMRate': rates[i]}, seed=0)
        times = dummy_model.t_domain()[:len(traj)]
        plt.plot(times, traj, label=str(rates[i]) + " kHz")

    plt.legend()
    plt.close()
    return


