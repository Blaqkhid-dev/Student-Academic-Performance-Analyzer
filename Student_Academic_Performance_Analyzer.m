clc;
clear;


% Student Academic Performance Analyzer

% Student names
names = ["Emmanuel" "Prosper" "Faruk" "Ini" "Niyi" "Akanbi" "Ade" "Ola" "Favour" "Mary"]

% Student Scores
scores = [65 72 81 54 90 68 77 43 88 71]

% Predictor variables
studyHours = [12, 15, 18, 8, 22, 10, 16, 5, 20, 14]; % hours spent studying
attendance = [85, 90, 95, 70, 98, 75, 88, 0, 96, 82]; % attendance parcentage(%)

% Calculate class stastistics
averageScore = mean(scores);
highestScore = max(scores);
lowestScore = min(scores);

% Clean and inspect data
numNames = length(names);
numScores = length(scores);
numHours = length(studyHours);
numAtt = length(attendance);

if (numNames == numScores) && (numScores == numHours) && (numHours == numAtt)
    disp('Data check passed: All feature lists match in length')
else
    disp('Data warning: Feature vector lengths do not match!')
end

% Check for scores outside the valid range
invalidScores = scores(scores < 0 | scores > 100);

if isempty(invalidScores)
    disp('Scores check passed: All scores are valid.')
else
    disp('Score Warning: Some scores are outside 0 and 100.')
end

% Additional statistics for model building
scoreRange = highestScore - lowestScore;
passingCount = sum(scores >= 50);




% Assign grades
grades = strings(size(scores));

for i = 1:length(scores)
    if scores(i) >= 70
        grades(i) = "A";
    elseif scores(i) >= 60
            grades(i) = "B";
    elseif scores(i) >= 50
        grades(i) = "C";
    elseif scores(i) >= 45
        grades(i) = "D";
    else 
        grades(i) = "F"
    end

end

% Display Student Results
fprintf('\nSTUDENT RESULT\n');
fprintf('--------------------\n')

for i = 1:length(names)
    fprintf('%-10s Scores: %3d  Grade: %s\n', names(i), scores(i), grades(i));
end

% Display class statistics
fprintf('\nCLASS STATISTICS\n')
fprintf('------------------\n')
fprintf('Class Average: %2f\n', averageScore);
fprintf('Highest Score: %2f\n', highestScore);
fprintf('Lowest Score: %2f\n', lowestScore);
fprintf('Score Range: %.2f\n', scoreRange);
fprintf('Passing Students: %d\n', passingCount);


% Visualise the data
figure;
bar(scores);
title('Student Academic Scores');
xlabel('Student Index');
ylabel('Scores');
xticklabels(names);
grid on;

% Create the predition model (Linear Regression)
% Combine predictor variables into a matrix (each row is a stdent)
X = [studyHours', attendance'];
y = scores';

% Calculate model coefficients using vector division
coefficients = X\y;
weightHours = coefficients(1);
weightAtt = coefficients(2);

% Create the baseline regression mode
X = [studyHours(:), attendance(:)];
y = scores(:);

% Solve for coefficients using matrix division
b = X\y;
disp('Baseline Coefficients [b_study; b_attendance]:')
disp(b)

X_intercept = [ones(size(y)), studyHours(:), attendance(:)];
b_full = X_intercept\y;

% Generate predictions and calculate metrics
y_pred = X * b;

% Calculate prediction errors (residuals)
residuals = y - y_pred;
mae = mean(abs(residuals)); %Mean Absolute error
fprintf('Mean Absoute Error (MAE): %.2f points\n', mae);

% Predict scores using our learned weights
predictedScores = X * coefficients;

% Test the model (Calculate errors)
actualVsPredicted = [scores', predictedScores];
mae = mean(abs(scores' - predictedScores));

% Display model insights
fprintf('\nPREDICTION MODE RESULTS\n')
fprintf('------------------\n')
fprintf('Study Hours Weight: %.2f points/hr\n', weightHours);
fprintf('Attendance Weight: %.2f points/%%\n', weightAtt);
fprintf('Mean Absoute Error: %.2f points\n', mae);

% Visualize model performance (Actual Vs Predicted)
% Prepare column vectors
actual = scores(:);
pred = predictedScores(:);

% Plot side-by-side comparison
figure;
bar([actual, pred]);

title('Actual Vs Predicted Student Score');
xlabel('Student Index');
ylabel('Scores');
legend('Actual Score', 'PredictedScore');
grid on;

% Visuaize Residuals (Prediction Errors)

% Calculate residuals (error for each student)
residuals = actual - pred;

% Plot residuals
figure;
plot(residuals, 'o-');

title('Prediction Residuals');
xlabel('Student Index');
ylabel('Residual (Actual - Predicted)');
grid on;

% Identify largest prediction errors

% Find the absolute prediction errors
absoluteErrors = abs(residuals);

% Find the largest error
[maxError, studentIndex] = max(absoluteErrors);

% Display the results
fprintf('\nLARGEST PREDICTION ERROR\n');
fprintf('------------------\n');
fprintf('Student: %s\n', names(studentIndex));
fprintf('Error: %.2f points\n', maxError);

% Final Mode Visualization
% Creating a dedicated figure window for model plots
figure('Name', 'Model Evaluation Visualizations', 'NumberTitle', 'off');

% Plot 1: Actual Vs Predicted Scores
subplot(1, 2, 1);
scatter(y, predictedScores, 60, 'filled', 'MarkerFaceColor', [0.2 0.4 0.8]);
hold on;

% Add reerence line for perfect prediction (y = x)
axisLimits = [min([y; predictedScores])-5, max([y; predictedScores])+5];
plot(axisLimits, axisLimits, 'r--', 'LineWidth', 1.5);

grid on;
xlim(axisLimits);
ylim(axisLimits);
title('Actual Vs Predicted Scores');
xlabel('Actual Scores');
ylabel('Predicted Scores');
legend('Student Data', 'Ideal Fit (y = x)', 'Location', 'northwest');
hold off;

% Plot 2: Residual/Error distribution
subplot(1, 2, 2);
histogram(residuals, 10, 'FaceColor', [0.8 0.3 0.3], 'EdgeColor', 'k');
grid on;
title('Residual Error Distribution');
xlabel('Prediction Error (Residuals)');
ylabel('Frequency');

% Reference Line at zero error
xline(0, 'k--', 'LineWidth', 1.5, 'Label', 'Zero Error');


% FINAL EVALUATION & EXPORT

% Calculate Evaluation Metrics (RMSE & R-squared)
n = length(y);
rmse = sqrt(mean(residuals.^2));
ss_tot = sum((y - mean(y)).^2);
ss_res = sum(residuals.^2);
r_squared = 1 - (ss_res / ss_tot);

% Final Model Evaluation Summary 
fprintf('\n=========================================\n');
fprintf('       FINAL MODEL EVALUATION SUMMARY     \n');
fprintf('=========================================\n');
fprintf('Root Mean Squared Error (RMSE) : %.2f points\n', rmse);
fprintf('Coefficient of Determination (R²): %.4f (%.1f%% variance explained)\n', r_squared, r_squared * 100);
fprintf('Mean Absolute Error (MAE)       : %.2f points\n', mae);
fprintf('Max Prediction Error            : %.2f points\n', maxError);
fprintf('=========================================\n');

%  Export Performance Results to CSV File
resultsTable = table(names(:), y(:), predictedScores(:), residuals(:), ...
    'VariableNames', {'Student', 'ActualScore', 'PredictedScore', 'Residual'});

writetable(resultsTable, 'Student_Performance_Results.csv');
fprintf('\nSuccess: Analysis results exported to "Student_Performance_Results.csv"\n');