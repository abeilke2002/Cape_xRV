from fpdf import FPDF
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
from datetime import datetime, timedelta
from great_tables import GT
import os
import numpy as np


Name = "Brown, Kade"

WIDTH = 210
HEIGHT = 297


def create_movement_plot(df, filename, name):
    player_data = df[df['Pitcher'] == name]

    fig, ax = plt.subplots(figsize=(10, 8)) 

    sns.scatterplot(data=player_data,
                    x='HorzBreak',
                    y='InducedVertBreak',
                    hue='TaggedPitchType',
                    s=150,
                    ax=ax)

    # Iterate through each pitch type
    for lab, col in zip(player_data['pitch_type'].unique(),
                        sns.color_palette(n_colors=len(player_data['pitch_type'].unique()))):

        xdata = player_data[player_data['pitch_type'] == lab]['HorzBreak']
        ydata = player_data[player_data['pitch_type'] == lab]['InducedVertBreak']

        try:
            # Calculate ellipse parameters if data is sufficient
            if len(xdata) > 1 and len(ydata) > 1:
                mean_x = np.mean(xdata)
                mean_y = np.mean(ydata)
                cov_matrix = np.cov(xdata, ydata)
                vals, vecs = np.linalg.eigh(cov_matrix)
                theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                width, height = 2 * 2 * np.sqrt(vals)

                # Create ellipse
                ell = Ellipse(xy=(mean_x, mean_y),
                              width=width, height=height,
                              angle=theta, color='black', alpha=0.2)
                ell.set_facecolor(col)
                ax.add_artist(ell)
        except Exception as e:
            print(f"Error calculating ellipse for pitch type {lab}: {e}")

    
    plt.axvline(x=0, color='black', linestyle='--')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    plt.xlabel('Horizontal Break', fontsize = 18)
    plt.ylabel('Induced Vertical Break', fontsize = 18)
    plt.title(f'Movement Plot for {player_data.iloc[0]["Pitcher"]}', fontsize = 18)
    plt.savefig(filename, transparent = True)  # Save the figure
    plt.close()

def create_time_plot(df, filename, name):
    player_data = df[df['Pitcher'] == name].copy()  # Make a copy to operate safely

    plt.figure(figsize=(15, 12))
    sns.lineplot(data=player_data, x='Inning', y='xRV', marker='o',ms = 30, 
                 hue='TaggedPitchType', color='deepskyblue', errorbar=None,
                 linewidth = 5)
    plt.title('xRV Variation Over The Course of Game', fontsize = 28)
    plt.xlabel('Inning', fontsize=28)
    plt.ylabel('xRV Measurement', fontsize=28)
    plt.tick_params(axis='y', labelsize=25)
    plt.tick_params(axis='x', labelsize=25)
    plt.grid(True)

    # Customize the legend
    plt.legend(title='Pitch Type', fontsize='16', title_fontsize='18')
    plt.savefig(filename, transparent = True)
    plt.close()

def create_table(df, filename, name):
    player_data = df[df['Pitcher'] == name]

    table = player_data.groupby('TaggedPitchType', as_index = False).agg({
        'Pitcher' : 'count',
        'RelSpeed' : 'mean',
        'InducedVertBreak' : 'mean',
        'HorzBreak' : 'mean',
        'SpinRate' : 'mean',
        'SpinAxis' : 'mean',
        'VertApprAngle' : 'mean',
        'HorzApprAngle' : 'mean',
        'RelHeight' : 'mean',
        'RelSide' : 'mean',
        'Extension' : 'mean',
        'xRV' : 'mean'
    }).rename(columns={
        'TaggedPitchType': 'Pitch Type',
        'Pitcher': 'Count',
        'RelSpeed': 'Velo',
        'InducedVertBreak': 'iVB',
        'HorzBreak': 'HB',
        'SpinRate': 'Spin Rate',
        'SpinAxis' : 'Spin Axis',
        'VertApprAngle': 'VAA',
        'HorzApprAngle': 'HAA',
        'RelHeight': 'vRel',
        'RelSide': 'hRel',
        'Extension': 'Ext',
        'xRV': 'xRV'
    }).round(2).sort_values(by = 'Count', ascending = False)

    table['xRV'] = table['xRV'].astype('float64').round()
    
    gt_tbl = GT(table)
    gt_tbl.save(filename)


def create_title(day, pdf):
    last_name, first_name = Name.split(', ')
    pdf.set_fill_color(255, 255, 255)  # Light blue background, for example
    pdf.rect(0, 0, WIDTH, HEIGHT, 'F')

    pdf.set_font('Arial', '', 24)
    pdf.ln(5)
    title = f"{first_name} {last_name} Stuff Report"
    title_width = pdf.get_string_width(title) 
    pdf.set_x((WIDTH - title_width) / 2)  
    pdf.write(10, title)
    pdf.ln(10)

    pdf.set_font('Arial', '', 16)
    date_str = f'Date: {day}'
    date_width = pdf.get_string_width(date_str)
    pdf.set_x((WIDTH - date_width) / 2)
    pdf.write(6, date_str)
    pdf.ln(10)

def create_report(day, filename=f"{Name.replace(' ', '_')}.pdf"):
    pdf = FPDF(unit='mm', format=(215, 200))
    pdf.add_page()
    create_title(day, pdf)

    half_width = WIDTH / 2 - 5
    full_width = WIDTH - 10

    base_directory = 'player_reports'
    date_directory = os.path.join(base_directory, str(day))  # Subfolder named after the date

    # Ensure the directory exists
    if not os.path.exists(date_directory):
        os.makedirs(date_directory)

    movement_plot_file = 'plots/movement_plot.png'
    create_movement_plot(df, movement_plot_file, Name)  # Adjusted to use dataframe `df` and corrected parameters
    pdf.image(movement_plot_file, x = 5, y = 40, w = half_width)

    # Generate and embed the time plot
    time_plot_file = 'plots/time_plot.png'
    create_time_plot(df, time_plot_file, Name)  # Adjusted to use dataframe `df` and corrected parameters
    pdf.image(time_plot_file, x=half_width + 10, y=40, w=half_width)

    stuff_table_file = 'plots/stuff_table.png'
    create_table(df, stuff_table_file, Name)
    pdf.image(stuff_table_file, x=5, y=125, w=full_width)
 

    directory = 'player_reports'
    if not os.path.exists(date_directory):
        os.makedirs(date_directory)

    # Save the file in the directory
    full_path = os.path.join(date_directory, filename)
    pdf.output(full_path, 'F')
    print(f"Report saved as {full_path}")  # Add this to confirm where the file is saved

if __name__ == '__main__':
    day = datetime.today().date()
    previous_day = day - timedelta(days=8) # If you want to do a specefic game update the "days =" to that game date. (days = 1 would be yesterday, days = 2 would be two days ago... etc)
    filtered_df = pd.read_csv("csvs/hyannis.csv")
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], format='%Y-%m-%d', errors='coerce')
    #df = filtered_df[filtered_df['Date'].dt.date == previous_day] # if this is commented out, then it will be the full season
    df = filtered_df
    create_report(day)
