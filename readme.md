# Description

This program is used to control the Optically Guided Tension Clamp (OGTC). Included here are 1) the pythoon code used to run the OGTC, 2) the sketches used to run the Teensy 4.1 miscocontroller, and 3) the reults from training the YOLOv8 object detection model.

# Getting Started

## Dependencies
Python 3.8.5 (other verison may work fine but it is not guaranteed)

## Installing

git clone https://github.com/GrandlLab/OGTC.git

Virtual Envuronment: pip install -r requirements.txt

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

# Authors
- Michael Sindoni: michael.sindoni@duke.edu

# License
The copyrights of this software are owned by Duke University. As such, two licenses for this software are offered: 
1. An open-source license under the CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/) license for non-commercial academic use.

2. A custom license with Duke University, for commercial use or uses without the CC BY-NC-ND 4.0 license restrictions.


As a recipient of this software, you may choose which license to receive the code under. Outside contributions to the Duke-owned code base cannot be accepted unless the contributor transfers the copyright to those changes over to Duke University.

To enter a custom license agreement without the CC BY-NC-ND 4.0 license restrictions, please contact the Digital Innovations department at the Duke Office for Translation & Commercialization (OTC) (https://otc.duke.edu/digital-innovations/#DI-team) at otcquestions@duke.edu with reference to “OTC File No. 8524” in your email.


Please note that this software is distributed AS IS, WITHOUT ANY WARRANTY; and without the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

