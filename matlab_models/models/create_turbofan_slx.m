%% JETWISE AI — Programmatic Simulink Turbofan Engine Model Builder
% Builds 'turbofan_engine_model.slx' with 9 interconnected turbomachinery subsystems.
% Requirements: MATLAB R2023a+ with Simulink installed.

fprintf('=== Building JETWISE AI Turbofan Simulink Model ===\n');

model_name = 'turbofan_engine_model';
slx_filename = [model_name '.slx'];

% Check Simulink availability
if ~license('test', 'SIMULINK')
    warning('Simulink license not available. Skipping SLX diagram generation.');
    return;
end

try
    % Close if already open
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end

    % Create new system
    new_system(model_name);
    open_system(model_name);

    % Configure solver parameters (Variable-step ode45 or Fixed-step ode4)
    set_param(model_name, 'Solver', 'ode45');
    set_param(model_name, 'StopTime', '10.0');

    % Add Inports for Operating Parameters
    add_block('simulink/Sources/In1', [model_name '/Throttle_Demand'], 'Position', [50, 100, 80, 114]);
    add_block('simulink/Sources/In1', [model_name '/Fuel_Flow_Demand'], 'Position', [50, 160, 80, 174]);
    add_block('simulink/Sources/In1', [model_name '/Ambient_Pressure'], 'Position', [50, 220, 80, 234]);
    add_block('simulink/Sources/In1', [model_name '/Ambient_Temp'], 'Position', [50, 280, 80, 294]);

    % Add Subsystems for 9 Turbofan Components
    components = {
        'Inlet_and_Fan', [150, 120, 250, 180];
        'LPC_Compressor', [300, 120, 400, 180];
        'HPC_Compressor', [450, 120, 550, 180];
        'Combustor', [600, 120, 700, 180];
        'HP_Turbine', [750, 120, 850, 180];
        'LP_Turbine', [900, 120, 1000, 180];
        'Exhaust_Nozzle', [1050, 120, 1150, 180];
        'HP_Shaft_Dynamics', [600, 220, 700, 270];
        'LP_Shaft_Dynamics', [600, 300, 700, 350]
    };

    for i = 1:size(components, 1)
        comp_name = components{i, 1};
        comp_pos = components{i, 2};
        add_block('simulink/Ports & Subsystems/Subsystem', [model_name '/' comp_name], 'Position', comp_pos);
    end

    % Add Outports for Multi-Station Sensors
    add_block('simulink/Sinks/Out1', [model_name '/Sensor_T24'], 'Position', [1200, 60, 1230, 74]);
    add_block('simulink/Sinks/Out1', [model_name '/Sensor_T30'], 'Position', [1200, 100, 1230, 114]);
    add_block('simulink/Sinks/Out1', [model_name '/Sensor_T50_EGT'], 'Position', [1200, 140, 1230, 154]);
    add_block('simulink/Sinks/Out1', [model_name '/Sensor_P30'], 'Position', [1200, 180, 1230, 194]);
    add_block('simulink/Sinks/Out1', [model_name '/Sensor_Nf_Fan_RPM'], 'Position', [1200, 220, 1230, 234]);
    add_block('simulink/Sinks/Out1', [model_name '/Sensor_Nc_Core_RPM'], 'Position', [1200, 260, 1230, 274]);

    % Save system
    save_system(model_name, slx_filename);
    close_system(model_name);

    fprintf('Simulink model built and saved successfully: %s\n', slx_filename);
catch ME
    fprintf('Error building Simulink model: %s\n', ME.message);
end
