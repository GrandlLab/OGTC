# Description

This program is used to control the Optically Guided Tension Clamp (OGTC). Included here are 1) the pythoon code used to run the OGTC, 2) the sketches used to run the Teensy 4.1 miscocontroller, and 3) the reults from training the YOLOv8 object detection model.

<div style="text-align: center;">
    <img title="a title" alt="Alt text" src="images/fig1_8-01_cut.jpg" style="width: 700px; height: auto;">
</div>

# Getting Started

## YOLOv8 training
- A YOLOv8 object detection model was trained on >1,500 Differential interference contrast microscopy (400x) images whereby the membrane dome was manually selected. The code for this training is provided in this repository as "training_code.py". This model can reliable detect the membrane done with confidence intervals at or greater then 0.7 as show by the training output below:

<div style="text-align: center;">
    <img title="a title" alt="Alt text" src="images/val_batch2_pred.jpg" style="width: 600px; height: auto;">
</div>

## Dependencies
Python 3.8.5 (other verison may work fine but it is not guaranteed)

## Installing the OGTC

git clone https://github.com/GrandlLab/OGTC.git

Virtual Envuronment: pip install -r requirements.txt

## Other Necessary Software

- **Micro-Manager**: This is an open-source software is used to control the microscope camera (Coolsnap EZ2) and can be downloaded at https://micro-manager.org/Download_Micro-Manager_Latest_Release

- **Arduino IDE**: This is an open-source software is used write and upload scripts to the Teensy 4.1 (code provided in the repository) and can be downloaded at https://www.arduino.cc/en/software


# Executing the Program

### Variables and Inputs

- **volts_per_bit**: slope of volts over bits calibration

- **volts_offset**: y-intercept of volts over bits calibration

- **bits_per_volt**: slope of bits over volts calibration

- **bits_offset**: y-intercept of volts over bits calibrationHEKA to teensy

- **command_voltage**: voltage program will begin at (0mmHg and 0V)

- **total_commanded_sweep_time**: total time you want each sweep to last; dictates how much rest will be given from sweep to sweep

- **target_tension_flex**: setting how flexible tension measurements from target tension can be before pressure is changed

- **hist_limits**: manually determining what the upper and lower bound on the pixel historgrams from the microscope images are

- **gaussian_column_num=2**: telling the OGTC program how many columns to skip when performing gaussian fitting

- **file_path**: file path to save images and end CSV

- **image_saving_frequency**: saving an image of the membrane and fit every x frames


# Program Output
- **protocol_data.csv**: a csv file containing all relevant data to the tension step protocol being executed. Descriptions for the data collected are below:

    - **monitor_bits**: bits being recorded by the Teensy 4.1. This value corresponds to the pressure that is present inside the patching pipette and is monitored by the high-speec pressure-clamp (HSPC) 

    - **command_bits**: bits being sent to the Teensy 4.1. This value corresponds to the pressure that is being commanded by the OGTC to the HSPC.

    - **monitor_pressure**: monitor_bits converted to the pressure (mmHg).

    - **target_tension**: tension being commanded for that step in the protocol.

    - **measured_tension**: tension that is present at the membrane.

    - **instant_radius**: calculated radius of the membrane dome for a single image.

    - **avg_radius**: calculated radius of the membrane dome for the five most recent images (not used for tension calculation).
    
    - **protocol_phase**: what phase the protocol is in (prepulse, pressure step, post pressure step).

    - **time**: total time the protocol has been running.

    - **sweep_time**: time for each individual tension step.

    - **on_off_heka**: recording if a voltage is being sent to the amplifier (1) or not (0). This is used as a timestamp for later analysis to timelock the start of the tension step to the electrophysiology recording.

    - **mem_fit_time**: times it takes for each individual loop in the OGTC program to occur.

- **fig_plot**: this variable in the code is associated with a JPG of the current DIC image with the membrane location in each pixel column (blue) and the circular fit (red) overlayed. An example image is shown below:

<div style="text-align: center;">
    <img title="a title" alt="Alt text" src="images/example_image_output.png" style="width: 400px; height: auto;">
</div>


# Authors
- Michael Sindoni: michael.sindoni@duke.edu

# License
The copyrights of this software are owned by Duke University. As such, two licenses for this software are offered: 
1. An open-source license under the CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/) license for non-commercial academic use.

2. A custom license with Duke University, for commercial use or uses without the CC BY-NC-ND 4.0 license restrictions.


As a recipient of this software, you may choose which license to receive the code under. Outside contributions to the Duke-owned code base cannot be accepted unless the contributor transfers the copyright to those changes over to Duke University.

To enter a custom license agreement without the CC BY-NC-ND 4.0 license restrictions, please contact the Digital Innovations department at the Duke Office for Translation & Commercialization (OTC) (https://otc.duke.edu/digital-innovations/#DI-team) at otcquestions@duke.edu with reference to “OTC File No. 8524” in your email.


Please note that this software is distributed AS IS, WITHOUT ANY WARRANTY; and without the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

