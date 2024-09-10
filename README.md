# Smart Coffee Machine Optimization Using AI & Deep Learning

## Project Overview

This project aims to optimize the energy efficiency of a coffee machine by leveraging deep learning algorithms and AI. The machine's heater is automatically turned off when it's not in use, reducing energy consumption without compromising user convenience.

## Data Collection

To gather real-world usage data, we placed a coffee machine in a public area at our university. Using **Node-RED**, we monitored and logged the following:

- When the machine is in use
- The type of coffee being brewed
- Machine power consumption patterns

The data was analyzed to determine peak usage times and energy consumption habits.

## Methodology

### 1. **Data Processing & Visualization**  
   The data was collected and visualized using Node-RED. Graphs of power consumption were plotted, and integrals of these graphs were computed to measure total energy usage. 

### 2. **AI for Coffee Type Detection**  
   We applied AI algorithms to identify which type of coffee is being produced. The model was trained to distinguish between:
   - Single-half cup
   - Single-full cup
   - Double-half cups
   - Double-full cups

   This classification allowed us to analyze coffee production patterns and adjust the machine's operation to optimize energy efficiency.

### 3. **Machine Learning & Deep Learning Models**  
   Using deep learning techniques, the system learned to predict idle times, allowing the heater to be turned off when the machine is not in use. This resulted in significant energy savings.

## Results

The implementation of AI and deep learning into the coffee machine system led to an improvement in energy efficiency by automating the heater's operation. By predicting when the machine would be idle, we reduced unnecessary heating periods.

## Tools & Technologies

- **Node-RED**: For data collection, visualization, and integral calculations
- **Deep Learning**: For AI-based predictions and classification of coffee types
- **Python/TensorFlow/Keras**: For machine learning model development

## Conclusion

This project demonstrates how AI and deep learning can be applied to real-world problems to create energy-efficient solutions. The smart coffee machine can predict user demand and adjust its operations accordingly, contributing to both energy savings and convenience.
