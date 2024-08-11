import pandas as pd 
import numpy as np
import pickle
import xgboost
import warnings

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

warnings.filterwarnings('ignore')

cape = pd.read_csv("/Users/aidanbeilke/Desktop/Postgame_pdfs/csvs/ml_ready.csv")
print(cape['pitch_type'].unique())

# Stuff Models
fb = "models/best_model_fb.pkl"
bb = "models/best_model_os.pkl"
os = "models/best_model_bb.pkl"

with open(fb, 'rb') as file:
    fb_model = pickle.load(file)

with open(bb, 'rb') as file:
    bb_model = pickle.load(file)

with open(os, 'rb') as file:
    os_model = pickle.load(file)

models = [fb_model, bb_model, os_model]


cols = ['Date', 'Pitcher', 'PitcherId', 'PitcherTeam', 'Batter', 'BatterId', 'Outs', 'Balls',
        'Strikes', 'TaggedPitchType', 'AutoPitchType', 'PitchCall','PlayResult' ,'RelSpeed',
        'VertRelAngle', 'HorzRelAngle', 'SpinRate', 'SpinAxis', 'Tilt', 'RelHeight',
        'RelSide', 'Extension', 'InducedVertBreak', 'HorzBreak', 'PlateLocHeight', 'PlateLocSide',
        'VertApprAngle', 'HorzApprAngle', 'BatterSide', 'PitcherThrows', 'Inning', 'PAofInning', 'pitch_type', 
        'ax0', 'ay0', 'az0', 'TaggedHitType']

rm_pitch = ['Undefined', 'Other', 'Knuckleball', 'Oneseamfastball']
remove = ['Undefined', 'StrkeSwinging', 'error', 'SIngle', 'homerun', 'BattersInterference', 'StriekC']
replacements = {
    'FoulBallNotFieldable': 'FoulBall',
    'FoulBallFieldable': 'FoulBall',
    'FouldBallNotFieldable' : 'FoulBall'
}

rv_dict = {
    'StrikeCalled' : -0.065092516,
    'StrikeSwinging' : -0.118124936,
    'BallCalled' : 0.063688329,
    'Out' : -0.195568767,
    'FoulBall' : -0.038050274,
    'Single' : 0.467292971,
    'FieldersChoice' : -0.195568767,
    'Triple' : 1.05755625,
    'Double' : 0.766083123,
    'SacBunt' : -0.10808108,
    'SacFly' : -0.236889646,
    'HomeRun' : 1.374328827,
    'Error' : -0.236889646,
    'HitByPitch' : 0.063688329,
}

result_match = [
    'StrikeCalled',
    'StrikeSwinging',
    'BallCalled',
    'Out',
    'FoulBall',
    'Single',
    'FieldersChoice',
    'Triple',
    'Double',
    'SacBunt',
    'SacFly',
    'HomeRun',
    'Error',
    'HitByPitch'
]



def classify_pitch_type(TaggedPitchType):
    if isinstance(TaggedPitchType, str):
        TaggedPitchType = TaggedPitchType.title()
    if TaggedPitchType in ['Fastball', 'Fourseamfastball']:
        return 'Fastball'
    elif TaggedPitchType in ['Twoseamfastball']:
        return 'Sinker'
    elif TaggedPitchType in ['ChangeUp']:
        return 'ChangeUp'  
    else:
        return TaggedPitchType
    
def primary_fastball(group):

    counts = group.value_counts()
    primary_pitch = counts.idxmax()
    return primary_pitch

fastball = ['Fastball', 'Sinker']
breaking = ['Curveball', 'Slider', 'Cutter']
offspeed = ['Splitter', 'ChangeUp']

features = [
    'RelSpeed', 
    'RelHeight', 
    'RelSide', 
    'Extension', 
    'HorzBreak', 
    'InducedVertBreak', 
    'Diff_Velocity', 
    'Diff_VerticalBreak', 
    'Diff_HorizontalBreak', 
    'ay0',
    'PitcherThrows'
]



def get_stuff_leaders(df):
    
    # limit data
    data = df[cols]
    data.loc[:, 'HorzBreakabs'] = data['HorzBreak'].abs()
    data.loc[:, 'RelSideabs'] = data['RelSide'].abs()
    
    # Primary Pitch
    df_filtered = data[data['TaggedPitchType'].isin(['Fastball', 'Sinker', 'Cutter'])].copy()
    df_filtered['PrimaryFastball'] = df_filtered.groupby('Pitcher')['TaggedPitchType'].transform(primary_fastball)
    
    primary_stats = df_filtered[df_filtered['TaggedPitchType'] == df_filtered['PrimaryFastball']].groupby('Pitcher').agg({
    'RelSpeed': 'mean',
    'InducedVertBreak': 'mean',
    'HorzBreak': 'mean',
    }).rename(columns=lambda x: 'Avg_' + x)
    
    data = data.merge(primary_stats, on='Pitcher', how='left')

    # Diff from primary
    data['Diff_Velocity'] = data['RelSpeed'] - data['Avg_RelSpeed']
    data['Diff_VerticalBreak'] = data['InducedVertBreak'] - data['Avg_InducedVertBreak']
    data['Diff_HorizontalBreak'] = data['HorzBreak'] - data['Avg_HorzBreak']

    data['PlayResult'] = data.apply(
        lambda row: 'SacBunt' if row['PlayResult'] == 'Sacrifice' and row['TaggedHitType'] == 'Bunt' 
        else ('SacFly' if row['PlayResult'] == 'Sacrifice' else row['PlayResult']),
        axis=1
    )
    
    data['result'] = np.where(data['PitchCall'] == 'InPlay', data['PlayResult'], data['PitchCall'])
    data = data[~data['result'].isin(remove)]
    data['result'] = data['result'].replace(replacements)
    
    data['run_value'] = data['result'].map(rv_dict)
    data = data[data['result'].isin(result_match)]  

    data['result_type_code'] = data['result'].astype('category').cat.codes

    print(data.loc[:, ['result', 'result_type_code']].drop_duplicates().sort_values(by = 'result_type_code'))
    data = data.drop_duplicates()

    df_fb = data[data['TaggedPitchType'].isin(fastball)]
    df_os = data[data['TaggedPitchType'].isin(offspeed)]
    df_bb = data[data['TaggedPitchType'].isin(breaking)]

    dfs = [df_fb, df_os, df_bb]
    dataframes_with_probs = []

    for model, df in zip(models, dfs):
        y_pred_prob = model.predict_proba(df[features])
        
        num_classes = y_pred_prob.shape[1]
        prob_columns = [f'prob_class_{i}' for i in range(num_classes)]
        prob_df = pd.DataFrame(y_pred_prob, columns=prob_columns)
        
        df = df.reset_index(drop=True)
        prob_df = prob_df.reset_index(drop=True)
        
        df_with_probs = pd.concat([df, prob_df], axis=1)
        
        # Calculate the value columns
        df_with_probs['bal_val'] = df_with_probs['prob_class_0'] * 0.063688329
        df_with_probs['do_val'] = df_with_probs['prob_class_1'] * 0.766083123
        df_with_probs['error_val'] = df_with_probs['prob_class_2'] * -0.236889646
        df_with_probs['fc_val'] = df_with_probs['prob_class_3'] * -0.195568767
        df_with_probs['fb_val'] = df_with_probs['prob_class_4'] * -0.038050274
        df_with_probs['hbp_val'] = df_with_probs['prob_class_5'] * 0.063688329
        df_with_probs['hmr_val'] = df_with_probs['prob_class_6'] * 1.374328827
        df_with_probs['out_val'] = df_with_probs['prob_class_7'] * -0.195568767
        df_with_probs['sacb_val'] = df_with_probs['prob_class_8'] * -0.10808108
        df_with_probs['sacf_val'] = df_with_probs['prob_class_9'] * -0.236889646
        df_with_probs['single_val'] = df_with_probs['prob_class_10'] * 0.467292971
        df_with_probs['sc_val'] = df_with_probs['prob_class_11'] * -0.065092516
        df_with_probs['sw_val'] = df_with_probs['prob_class_12'] * -0.118124936
        df_with_probs['triple_val'] = df_with_probs['prob_class_13'] * 1.05755625
        
        df_with_probs['rv_val'] = df_with_probs[['bal_val', 'do_val', 'error_val', 'fc_val', 'fb_val', 'hbp_val', 'hmr_val', 'out_val', 'sacb_val', 'sacf_val', 'single_val', 'sc_val', 'sw_val', 'triple_val']].sum(axis=1)
        
        dataframes_with_probs.append(df_with_probs)

    fb_stuff, os_stuff, bb_stuff = dataframes_with_probs

    fb_stuff['xRV'] = 100 - ((fb_stuff['rv_val'] - fb_stuff['rv_val'].mean()) / fb_stuff['rv_val'].std()) * 10
    os_stuff['xRV'] = 100 - ((os_stuff['rv_val'] - os_stuff['rv_val'].mean()) / os_stuff['rv_val'].std()) * 10
    bb_stuff['xRV'] = 100 - ((bb_stuff['rv_val'] - bb_stuff['rv_val'].mean()) / bb_stuff['rv_val'].std()) * 10

    combined_df = pd.concat([fb_stuff, os_stuff, bb_stuff], ignore_index= True).drop_duplicates()
    
    return combined_df

check = get_stuff_leaders(cape)
check['Date'] = pd.to_datetime(check['Date'], format='%Y-%m-%d', errors='coerce')

check_df = pd.DataFrame(check)

check_df.to_csv('csvs/hyannis.csv', index=False)
print("Data saved to csvs/hyannis.csv")

