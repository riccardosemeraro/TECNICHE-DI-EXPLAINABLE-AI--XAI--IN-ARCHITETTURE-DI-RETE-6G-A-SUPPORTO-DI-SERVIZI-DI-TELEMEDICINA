%% ------------------------------------------------------------------------
%IMPLEMENTAZIONE PCA-------------------------------------------------------
%--------------------------------------------------------------------------

% Caricamento del dataset
diabetes_prediction = readtable("diabetes_prediction_dataset_stringToInt.csv");

% Estrazione delle feature e conversione della tabella in una matrice
X = diabetes_prediction(:, 1:end-1);
feature_names = X.Properties.VariableNames;

disp("Dataset:")
disp(X(1:5, :))

disp("Feature Names:")
disp(feature_names)

% Conversione della tabella in una matrice per l'analisi
X_matrix = table2array(X);

disp("Matrice per analisi")
disp(X_matrix(1:5,:))

% Esecuzione PCA
[coeff, score, latent, tsquared, explained, mu] = pca(X_matrix);

% Calcolo della varianza cumulativa
cumulativeExplained = cumsum(explained);
disp("Varianza cumulativa: ")
disp(cumulativeExplained)
%%

% Determinazione del numero di componenti che spiegano almeno il 95% della varianza
numComponents95 = find(cumulativeExplained >= 95, 1);
disp(['Numero di Componenti Principali: ', num2str(numComponents95)]);

%stampa score
disp("SCORE:");
disp(score(1:5,:))


% Utilizzo di questo numero di componenti per ridurre il dataset
X_reduced = score(:, 1:numComponents95);
disp(['Number of components used: ', num2str(numComponents95)]);
disp('Reduced dimension dataset:');
disp(X_reduced(1:5,:)); %score reduced

[rows, cols] = size(X_reduced);  % Ottieni numero di righe e colonne
disp(['Numero di righe X_reduced: ', num2str(rows)])
disp(['Numero di colonne X_reduced: ', num2str(cols)])

intestazioni = {};
for i = 1:1:numComponents95
    intestazioni{i} = sprintf('PC%d', i);
end

disp(intestazioni)

% Aggiunta dell'ultima colonna (target) al dataset ridotto
target_column = diabetes_prediction(:, end);
X_reduced_principal_components_with_target = [array2table(X_reduced, 'VariableNames', intestazioni), target_column];

% Salvataggio della tabella in un file CSV
writetable(X_reduced_principal_components_with_target,'reduced_diabetes_principal_components_cumulative_explained_95.csv');

%%
% Determinazione del numero di componenti che spiegano almeno il 99% della varianza
numComponents99 = find(cumulativeExplained >= 99, 1);
disp(['Numero di Componenti Principali: ', num2str(numComponents)]);

% Utilizzo di questo numero di componenti per ridurre il dataset
X_reduced = score(:, 1:numComponents99);
disp(['Number of components used: ', num2str(numComponents99)]);
disp('Reduced dimension dataset:');
disp(X_reduced(1:5,:)); %score reduced

[rows, cols] = size(X_reduced);  % Ottieni numero di righe e colonne
disp(['Numero di righe X_reduced: ', num2str(rows)])
disp(['Numero di colonne X_reduced: ', num2str(cols)])

intestazioni = {};
for i = 1:1:numComponents99
    intestazioni{i} = sprintf('PC%d', i);
end

disp(intestazioni)

% Aggiunta dell'ultima colonna (target) al dataset ridotto
target_column = diabetes_prediction(:, end);
X_reduced_principal_components_with_target = [array2table(X_reduced, 'VariableNames', intestazioni), target_column];

% Salvataggio della tabella in un file CSV
writetable(X_reduced_principal_components_with_target,'reduced_diabetes_principal_components_cumulative_explained_99.csv');
%%
%-------------------GRAFICO PARETO--------------------------------

% Grafico Pareto
figure(10);
pareto(explained, intestazioni);
title('Grafico di Pareto della Varianza Spiegata dalle Componenti Principali');
xlabel('Componenti Principali');
ylabel('Varianza Spiegata (%)');
saveas(gcf, 'grafico_pareto.png');  % Salva la figura come PNG

%%
%-----------------------------PROVA PER VISUALIZZARE I NOMI DELLE FEATURE PRINCIPALI DELLA PRIMA E SECONDA COMPONENTE-----------------------

% Trovare le feature che influenzano maggiormente la prima e la seconda componente principale
[~, idx_first_component] = sort(abs(coeff(:,1)), 'descend');  % Ordinare i contributi della prima componente
%disp(coeff(:,:));
[~, idx_second_component] = sort(abs(coeff(:,2)), 'descend'); % Ordinare i contributi della seconda componente

[~, idx_third_component] = sort(abs(coeff(:,3)), 'descend'); % Ordinare i contributi della terza componente

% Visualizzare i nomi delle feature principali per la prima componente
disp('1. Feature principali per la prima componente principale:')
disp(feature_names(idx_first_component(1:4)))  % Mostra le prime 4 feature con il contributo maggiore

% Visualizzare i nomi delle feature principali per la seconda componente
disp('2. Feature principali per la seconda componente principale:')
disp(feature_names(idx_second_component(1:4)))  % Mostra le prime 4 feature con il contributo maggiore

% Visualizzare i nomi delle feature principali per la terza componente
disp('3. Feature principali per la terza componente principale:')
disp(feature_names(idx_third_component(1:4)))  % Mostra le prime 4 feature con il contributo maggiore


% Aprire un file per la scrittura (modifica il nome del file se necessario)
fileID = fopen('feature_principali_per_prima_seconda_terza_componente.txt', 'w'); % 'w' per scrivere
% Verificare se il file è stato aperto correttamente
if fileID == -1
    error('Non è stato possibile aprire il file.');
end
% Visualizzare e scrivere i nomi delle feature principali per la prima componente
fprintf(fileID, '1. Feature principali per la prima componente principale:\n');
fprintf(fileID, '%s\n', feature_names{idx_first_component(1:4)});  % Mostra le prime 4 feature con il contributo maggiore
% Visualizzare e scrivere i nomi delle feature principali per la seconda componente
fprintf(fileID, '2. Feature principali per la seconda componente principale:\n');
fprintf(fileID, '%s\n', feature_names{idx_second_component(1:4)});  % Mostra le prime 4 feature con il contributo maggiore
% Visualizzare e scrivere i nomi delle feature principali per la terza componente
fprintf(fileID, '3. Feature principali per la terza componente principale:\n');
fprintf(fileID, '%s\n', feature_names{idx_third_component(1:4)});  % Mostra le prime 4 feature con il contributo maggiore
% Chiudere il file
fclose(fileID);
% Confermare che l'output è stato salvato
disp('Output salvato in output.txt');

%% ------------------------------------------------------------------------
%SELEZIONE FEATURES MENO RILEVANTI CON PCA-CUMULATIVE EXPLAINED 95---------
%--------------------------------------------------------------------------

% Identificazione delle feature meno rilevanti con cumulative explained 95
% Selezione solo dei primi 'numComponents'
selected_coeff = coeff(:, 1:numComponents95);

% Calcolo della somma dei contributi di ogni feature ai componenti principali selezionati
feature_contributions = sum(abs(selected_coeff), 2);

% Numero di feature da mantenere
threshold = median(feature_contributions); % Uso della mediana come soglia

% feature con contributi sopra la soglia
features_to_keep = feature_contributions >= threshold;

% Riduzione del dataset originale mantenendo solo le feature rilevanti
X_relevant_features = X_matrix(:, features_to_keep);
relevant_feature_names = feature_names(features_to_keep);

disp("Nomi Feature Rilevanti Trovati con la Cumulative Explained 95")
disp(relevant_feature_names)

% Aggiunta dell'ultima colonna (target) al dataset ridotto
target_column = diabetes_prediction(:, end);
X_relevant_features_with_target = [array2table(X_relevant_features, 'VariableNames', relevant_feature_names), target_column];

% Visualizzazione del dataset ridotto con le feature rilevanti e la colonna target
disp('Dataset with relevant features and target column:');
disp(X_relevant_features_with_target(1:5,:));

% Salvataggio della tabella in un file CSV
writetable(X_relevant_features_with_target,'reduced_diabetes_prediction_features_with_target_cumulative_explained_95.csv');

%% ------------------------------------------------------------------------
%SELEZIONE FEATURES MENO RILEVANTI CON PCA-CUMULATIVE EXPLAINED 99---------
%--------------------------------------------------------------------------

% Identificazione delle feature meno rilevanti con cumulative explained 99
% Selezione solo dei primi 'numComponents'
selected_coeff = coeff(:, 1:numComponents99);
disp("TIPOOOO")
disp(class(selected_coeff))

% Calcolo della somma dei contributi di ogni feature ai componenti principali selezionati
feature_contributions = sum(abs(selected_coeff), 2);

% Numero di feature da mantenere
threshold = median(feature_contributions); % Uso della mediana come soglia

% feature con contributi sopra la soglia
features_to_keep = feature_contributions >= threshold;

% Riduzione del dataset originale mantenendo solo le feature rilevanti
X_relevant_features = X_matrix(:, features_to_keep);
relevant_feature_names = feature_names(features_to_keep);

disp("Nomi Feature Rilevanti Trovati con la Cumulative Explained 99")
disp(relevant_feature_names)

% Aggiunta dell'ultima colonna (target) al dataset ridotto
target_column = diabetes_prediction(:, end);
X_relevant_features_with_target = [array2table(X_relevant_features, 'VariableNames', relevant_feature_names), target_column];

% Visualizzazione del dataset ridotto con le feature rilevanti e la colonna target
disp('Dataset with relevant features and target column:');
disp(X_relevant_features_with_target(1:5,:));

% Salvataggio della tabella in un file CSV
writetable(X_relevant_features_with_target,'reduced_diabetes_prediction_features_with_target_cumulative_explained_99.csv');

%% ------------------------------------------------------------------------
%SCATTER PLOT 2D TRA PRIMA E SECONDA COMPONENT-----------------------------
%--------------------------------------------------------------------------

% Creare una figura
target_column = diabetes_prediction(:, end);
target = table2array(target_column);
% Visualizzazione dello scatter plot delle prime due componenti principali
figure(2);
% Creazione dello scatter plot
scatter(score(:, 1), score(:, 2), 10, target, 'filled');
% Impostazione delle etichette degli assi con i nomi delle feature e le percentuali di varianza
xlabel(['Componente Principale 1 (' num2str(explained(1), '%.2f') '%)'], 'FontSize', 12);
ylabel(['Componente Principale 2 (' num2str(explained(2), '%.2f') '%)'], 'FontSize', 12);
title('Scatter Plot 2D delle Prime Due Componenti Principali');
grid on;
colorbar; % Aggiunge una barra dei colori per mostrare le classi

% Aggiungere le equazioni per le componenti principali
text(0.1, 0.97, '$PC_1 = w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3$', ...
    'Interpreter', 'latex', 'FontSize', 12, 'Units', 'normalized');
text(0.1, 0.92, '$PC_2 = w_4 \cdot x_1 + w_5 \cdot x_2 + w_6 \cdot x_3$', ...
    'Interpreter', 'latex', 'FontSize', 12, 'Units', 'normalized');

saveas(gcf, 'scatter_plot_prime_due_componenti_principali.png');  % Salva la figura come PNG

%% ------------------------------------------------------------------------
%SCATTER PLOT 3D TRA PRIMA, SECONDA E TERZA COMPONENTE---------------------
%--------------------------------------------------------------------------

% Creare una figura
target_column = diabetes_prediction(:, end);
target = table2array(target_column);
% Visualizzazione dello scatter plot delle prime due componenti principali
figure(3);
% Creazione dello scatter plot
scatter3(score(:, 1), score(:, 2), score(:,3), 10, target, 'filled');
% Impostazione delle etichette degli assi con i nomi delle feature e le percentuali di varianza
xlabel(['Componente Principale 1 (' num2str(explained(1), '%.2f') '%)'], 'FontSize', 12);
ylabel(['Componente Principale 2 (' num2str(explained(2), '%.2f') '%)'], 'FontSize', 12);
zlabel(['Componente Principale 3 (' num2str(explained(3), '%.2f') '%)'], 'FontSize', 12);
title('Scatter Plot 3D delle Prime Tre Componenti Principali');
grid on;
colorbar; % Aggiunge una barra dei colori per mostrare le classi

% Aggiungere le equazioni per le componenti principali
text(0.1, 0.95, '$PC_1 = w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3$', ...
    'Interpreter', 'latex', 'FontSize', 12, 'Units', 'normalized');
text(0.1, 0.90, '$PC_2 = w_4 \cdot x_1 + w_5 \cdot x_2 + w_6 \cdot x_3$', ...
    'Interpreter', 'latex', 'FontSize', 12, 'Units', 'normalized');
text(0.1, 0.85, '$PC_3 = w_7 \cdot x_1 + w_8 \cdot x_2 + w_9 \cdot x_3$', ...
    'Interpreter', 'latex', 'FontSize', 12, 'Units', 'normalized');

saveas(gcf, 'scatter_plot_prime_tre_componenti_principali.png');  % Salva la figura come PNG

%% ------------------------------------------------------------------------
% Plot Formule Matematiche delle Componenti Principali


figure(4);

% Impostare il nome e la posizione
set(gcf, 'Name', 'Formule Componenti Principali 1:3', 'Position', [100, 100, 1000, 200]);

axis off;
hold on;

formula = ['$PC_1 = w_1 \cdot ' feature_names{idx_first_component(1)} ...
    ' + w_2 \cdot ' feature_names{idx_first_component(2)} ...
    ' + w_3 \cdot ' feature_names{idx_first_component(3)} ...
    ' + w_4 \cdot ' feature_names{idx_first_component(4)} '$'];


text(0.1, 0.9, formula, 'Interpreter', 'latex', 'FontSize', 24);

fprintf(formula)

formula = ['$PC_2 = w_1 \cdot ' feature_names{idx_second_component(1)} ...
    ' + w_2 \cdot ' feature_names{idx_second_component(2)} ...
    ' + w_3 \cdot ' feature_names{idx_second_component(3)} ...
    ' + w_4 \cdot ' feature_names{idx_second_component(4)} '$'];

text(0.1, 0.7, formula, 'Interpreter', 'latex', 'FontSize', 24);

fprintf(formula)

formula = ['$PC_3 = w_1 \cdot ' feature_names{idx_third_component(1)} ...
    ' + w_2 \cdot ' feature_names{idx_third_component(2)} ...
    ' + w_3 \cdot ' feature_names{idx_third_component(3)} ...
    ' + w_4 \cdot ' feature_names{idx_third_component(4)} '$'];

text(0.1, 0.5, formula, 'Interpreter', 'latex', 'FontSize', 24);

fprintf(formula)

saveas(gcf, 'formule_prime_tre_componenti_principali.png');  % Salva la figura come PNG