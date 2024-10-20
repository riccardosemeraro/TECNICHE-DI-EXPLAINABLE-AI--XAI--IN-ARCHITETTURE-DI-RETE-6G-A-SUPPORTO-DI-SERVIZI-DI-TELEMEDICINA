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

% Determinazione del numero di componenti che spiegano almeno il 95% della varianza
numComponents = find(cumulativeExplained >= 95, 1);

%stampa score
disp("SCORE:");
disp(score(1:5,:))

% Utilizzo di questo numero di componenti per ridurre il dataset
X_reduced = score(:, 1:numComponents);
disp(['Number of components used: ', num2str(numComponents)]);
disp('Reduced dimension dataset:');
disp(X_reduced(1:5,:)); %score reduced

%-------------------GRAFICO PARETO--------------------------------

% Grafico Pareto
figure(10);
pareto(explained);
title('Grafico di Pareto della Varianza Spiegata dalle Componenti Principali');
xlabel('Componenti Principali');
ylabel('Varianza Spiegata (%)');

%-----------------------------PROVA PER VISUALIZZARE I NOMI DELLE FEATURE PRINCIPALI DELLA PRIMA E SECONDA COMPONENTE-----------------------

% Trovare le feature che influenzano maggiormente la prima e la seconda componente principale
[~, idx_first_component] = sort(abs(coeff(:,1)), 'descend');  % Ordinare i contributi della prima componente
disp(coeff(:,:));
[~, idx_second_component] = sort(abs(coeff(:,2)), 'descend'); % Ordinare i contributi della seconda componente

% Visualizzare i nomi delle feature principali per la prima componente
disp('Feature principali per la prima componente principale:')
disp(feature_names(idx_first_component(1:4)))  % Mostra le prime 4 feature con il contributo maggiore

% Visualizzare i nomi delle feature principali per la seconda componente
disp('Feature principali per la seconda componente principale:')
disp(feature_names(idx_second_component(1:4)))  % Mostra le prime 4 feature con il contributo maggiore

%% ------------------------------------------------------------------------
%SELEZIONE FEATURES MENO RILEVANTI CON PCA---------------------------------
%--------------------------------------------------------------------------

% Identificazione delle feature meno rilevanti
% Selezione solo dei primi 'numComponents'
selected_coeff = coeff(:, 1:numComponents);

% Calcolo della somma dei contributi di ogni feature ai componenti principali selezionati
feature_contributions = sum(abs(selected_coeff), 2);

% Numero di feature da mantenere
threshold = median(feature_contributions); % Uso della mediana come soglia

% feature con contributi sopra la soglia
features_to_keep = feature_contributions >= threshold;

% Riduzione del dataset originale mantenendo solo le feature rilevanti
X_relevant_features = X_matrix(:, features_to_keep);
relevant_feature_names = feature_names(features_to_keep);

% Aggiunta dell'ultima colonna (target) al dataset ridotto
target_column = diabetes_prediction(:, end);
X_relevant_features_with_target = [array2table(X_relevant_features, 'VariableNames', relevant_feature_names), target_column];

% Visualizzazione del dataset ridotto con le feature rilevanti e la colonna target
disp('Dataset with relevant features and target column:');
disp(X_relevant_features_with_target(1:5,:));

% Salvataggio della tabella in un file CSV
writetable(X_relevant_features_with_target,'reduced_diabetes_prediction_features_with_target.csv');

%% ------------------------------------------------------------------------







%RANDOM FOREST DOPO PCA E ACCURATEZZA E METRICHE PERFORMANCE---------------
%--------------------------------------------------------------------------
disp(' ');
disp("RANDOM FOREST DOPO PCA")

% Conversione della colonna target in un array (se non lo è già)
target = table2array(target_column);

% Divisione del dataset in training e test set (ad esempio, 80% training, 20% test)
cv = cvpartition(target, 'Holdout', 0.3);
X_train = X_relevant_features(cv.training, :);
y_train = target(cv.training);
X_test = X_relevant_features(cv.test, :);
y_test = target(cv.test);

% Numero di campioni nel test set
num_test_samples = sum(cv.test); % Somma degli elementi del test set
disp(['Numero di campioni nel test set: ', num2str(num_test_samples)]);

% Creazione della Random Forest con 100 alberi
numTrees = 100;
randomForestModel = TreeBagger(numTrees, X_train, y_train, ...
    'OOBPrediction', 'On', ...
    'OOBPredictorImportance', 'on', ... % Abilita l'importanza delle feature
    'Method', 'classification');

% Previsione sul test set
predictions = predict(randomForestModel, X_test);

% Convertire le previsioni da cell array a un array numerico
predictions = str2double(predictions);

% Calcolo dell'accuratezza
accuracy = sum(predictions == y_test) / length(y_test);
disp(['Test Set Accuracy: ', num2str(accuracy)]);

% Calcolo delle metriche di classificazione
classes = unique(y_test); % Uniche classi
num_classes = length(classes);
precision = zeros(num_classes, 1);
recall = zeros(num_classes, 1);
f1_score = zeros(num_classes, 1);
support = zeros(num_classes, 1);

for i = 1:num_classes
    % Indici per la classe corrente
    true_positives = sum((y_test == classes(i)) & (predictions == classes(i)));
    false_positives = sum((y_test ~= classes(i)) & (predictions == classes(i)));
    false_negatives = sum((y_test == classes(i)) & (predictions ~= classes(i)));
    
    % Calcolo precision, recall, f1-score e supporto
    precision(i) = true_positives / (true_positives + false_positives + eps); % eps per evitare divisione per zero
    recall(i) = true_positives / (true_positives + false_negatives + eps);
    f1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
    %support(i) = sum(y_test == classes(i));
    support(i) = num_test_samples/2;
end

% Calcolo delle medie
macro_avg_precision = mean(precision);
macro_avg_recall = mean(recall);
macro_avg_f1_score = mean(f1_score);
%macro_avg_support = mean(support);
macro_avg_support = num_test_samples;

weighted_avg_precision = sum(precision .* support) / sum(support);
weighted_avg_recall = sum(recall .* support) / sum(support);
weighted_avg_f1_score = sum(f1_score .* support) / sum(support);
%weighted_avg_support = sum(support); % support è totale
weighted_avg_support = num_test_samples;

% Inizializzazione delle colonne come stringhe normali
column1 = {}; % Colonna delle classi
column2 = {}; % Precisione
column3 = {}; % Richiamo
column4 = {}; % F1-Score
column5 = {}; % Supporto

% Loop attraverso le classi
for i = 1:num_classes
    column1 = [column1; sprintf('Class %d', classes(i))]; % Nome della classe
    column2 = [column2; num2str(precision(i))]; % Precisione per ogni classe
    column3 = [column3; num2str(recall(i))]; % Richiamo per ogni classe
    column4 = [column4; num2str(f1_score(i))]; % F1-Score per ogni classe
    column5 = [column5; num2str(support(i))]; % Supporto per ogni classe
end

column1 = [column1; ' ']; 
column2 = [column2; ' ']; 
column3 = [column3; ' ']; 
column4 = [column4; ' ']; 
column5 = [column5; ' ']; 

% Aggiungere le righe per Accuracy, Macro Avg e Weighted Avg
column1 = [column1; 'Accuracy'; 'Macro Avg'; 'Weighted Avg'];
column2 = [column2; ' '; num2str(macro_avg_precision); num2str(weighted_avg_precision)];
column3 = [column3; ' '; num2str(macro_avg_recall); num2str(weighted_avg_recall)];
column4 = [column4; num2str(accuracy); num2str(macro_avg_f1_score); num2str(weighted_avg_f1_score)];
column5 = [column5; num2str(num_test_samples); num2str(macro_avg_support); num2str(weighted_avg_support)];

% Creazione della tabella finale con le colonne
tableClassificationReport = table(column1, column2, column3, column4, column5, ...
    'VariableNames', {' ', 'Precision', 'Recall', 'F1-Score', 'Support'});

% Visualizzazione della tabella
disp(' ');
disp('Classification Report: ');
disp(' ');
disp(tableClassificationReport);

% % Visualizzazione dell'accuratezza OOB
% % oobErrorBaggedEnsemble = oobError(randomForestModel);
% % figure;
% % plot(oobErrorBaggedEnsemble);
% % xlabel('Number of grown trees');
% % ylabel('Out-of-bag classification error');
% % title('Out-of-bag Error');

% Calcolo dell'importanza delle feature
featureImportance = randomForestModel.OOBPermutedPredictorDeltaError;

% Visualizzazione dell'importanza delle feature
figure(1);
bar(featureImportance);
xlabel('Feature Important Names');
ylabel('Importance Score');
title('Feature Importance for Random Forest');
% Aggiungi i nomi delle feature sull'asse x
set(gca, 'XTickLabel', relevant_feature_names, 'XTick', 1:numel(featureImportance));
xtickangle(0); % Ruota le etichette per una migliore leggibilità

%% Recupera i nomi delle feature più importanti per le prime quattro componenti andandole ad ordinare

disp(relevant_feature_names)
disp(featureImportance)

% Ordinare i punteggi e ottenere gli indici di ordinamento
[sorted_featureImportance, sorted_indices] = sort(featureImportance, 'descend');

% Riordinare i nomi delle feature secondo l'ordinamento dei punteggi
sorted_relevant_feature_names = relevant_feature_names(sorted_indices);

% Visualizzare i risultati
disp('Nomi delle feature ordinati:');
disp(sorted_relevant_feature_names);
disp('Score ordinati:');
disp(sorted_featureImportance);

%%
% Visualizzazione dello scatter plot 3D delle prime tre componenti principali
figure(2);
% Creazione dello scatter plot
scatter3(score(:, 1), score(:, 2), score(:, 3), 10, target, 'filled');
% Recupero delle percentuali di varianza spiegata per le prime tre componenti
var1 = explained(1); % Percentuale per la prima componente
var2 = explained(2); % Percentuale per la seconda componente
var3 = explained(3); % Percentuale per la terza componente
% Nomi delle feature originali per le prime tre componenti principali
feature_names_1 = sorted_relevant_feature_names(1); % Nomi delle feature per la prima componente
feature_names_2 = sorted_relevant_feature_names(2); % Nomi delle feature per la seconda componente
feature_names_3 = sorted_relevant_feature_names(3); % Nomi delle feature per la terza componente
% Impostazione delle etichette degli assi con i nomi delle feature e le percentuali di varianza
xlabel(['Componente Principale 1 (', num2str(var1, '%.2f'), '%) - ', strjoin(feature_names_1', ', ')]);
ylabel(['Componente Principale 2 (', num2str(var2, '%.2f'), '%) - ', strjoin(feature_names_2', ', ')]);
zlabel(['Componente Principale 3 (', num2str(var3, '%.2f'), '%) - ', strjoin(feature_names_3', ', ')]);
title('Scatter Plot 3D delle Prime Tre Componenti Principali');
grid on;
colorbar; % Aggiunge una barra dei colori per mostrare le classi

%%
% ------------------------------------------------------------------------
% Visualizzazione dello scatter plot a coppie delle prime quattro componenti principali
% ------------------------------------------------------------------------

% Selezione delle prime quattro componenti principali
numComponents = 4; % Cambia a 4 per le prime quattro componenti
X_four_components = score(:, 1:numComponents);

% Creazione dello scatter plot a coppie
figure(3);
gplotmatrix(X_four_components, [], [], 'b', 'o', 5, 'on');
te
% Recupero delle percentuali di varianza spiegata per le prime quattro componenti
var1 = explained(1); % Percentuale per la prima componente
var2 = explained(2); % Percentuale per la seconda componente
var3 = explained(3); % Percentuale per la terza componente
var4 = explained(4); % Percentuale per la quarta componente

% Impostazione del titolo
sgtitle('Scatter Plot a Coppie delle Prime Quattro Componenti Principali');

% Impostazione delle etichette degli assi
% Le etichette degli assi saranno impostate automaticamente da gplotmatrix.


%%
% ------------------------------------------------------------------------
% Visualizzazione dello scatter plot 4D delle prime quattro componenti principali
% ------------------------------------------------------------------------

% Selezione delle prime quattro componenti principali
numComponents = 4; % Cambia a 4 per le prime quattro componenti
X_four_components = score(:, 1:numComponents);

% Creazione dello scatter plot 3D
figure(4);
scatter3(X_four_components(:, 1), X_four_components(:, 2), X_four_components(:, 3), ...
    50, X_four_components(:, 4), 'filled'); % Usa la quarta componente per colorare i punti

% Recupero delle percentuali di varianza spiegata per le prime quattro componenti
var1 = explained(1); % Percentuale per la prima componente
var2 = explained(2); % Percentuale per la seconda componente
var3 = explained(3); % Percentuale per la terza componente
var4 = explained(4); % Percentuale per la quarta componente

% Nomi delle feature originali per le prime quattro componenti principali
feature_names_1 = sorted_relevant_feature_names(1); % Nomi delle feature per la prima componente
feature_names_2 = sorted_relevant_feature_names(2); % Nomi delle feature per la seconda componente
feature_names_3 = sorted_relevant_feature_names(3); % Nomi delle feature per la terza componente
feature_names_4 = sorted_relevant_feature_names(4); % Nomi delle feature per la quarta componente

% Impostazione delle etichette degli assi con i nomi delle feature e le percentuali di varianza
xlabel(['Componente Principale 1 (', num2str(var1, '%.2f'), '%) - ', strjoin(feature_names_1', ', ')]);
ylabel(['Componente Principale 2 (', num2str(var2, '%.2f'), '%) - ', strjoin(feature_names_2', ', ')]);
zlabel(['Componente Principale 3 (', num2str(var3, '%.2f'), '%) - ', strjoin(feature_names_3', ', ')]);
title('Scatter Plot 3D delle Prime Quattro Componenti Principali');
grid on;

% Aggiunta di una barra dei colori per mostrare la quarta componente
cbar = colorbar; % Crea la barra dei colori
caxis([min(X_four_components(:, 4)), max(X_four_components(:, 4))]); % Normalizza la scala dei colori

% Imposta la label per la barra dei colori
ylabel(cbar, ['Componente Principale 4 (', num2str(var4, '%.2f'), '%) - ', strjoin(feature_names_4', ', ')]); % Aggiungi la label alla barra dei colori

% Imposta un colormap (opzionale)
colormap(jet); % Puoi cambiare 'jet' in un'altra colormap se preferisci

%%
% Selezione delle prime quattro componenti principali
numComponents = 4; % Cambia a 4 per le prime quattro componenti
X_four_components = score(:, 1:numComponents);

% Visualizzazione del matrix plot delle prime quattro componenti
figure(5);
ax = gplotmatrix(X_four_components, [], target, [], [], [], [], "grpbars");
title("Matrix Plot delle Prime Quattro Componenti Principali");


% Aggiunta di una legenda
hold on; % Mantieni il grafico
unique_classes = unique(target);
legend_entries = arrayfun(@(c) sprintf('Classe %d', c), unique_classes, 'UniformOutput', false);

% Creare un segnaposto per la legenda
for i = 1:length(unique_classes)
    plot(nan, nan, 'o', 'DisplayName', legend_entries{i});
end

%%
% Selezione delle prime quattro componenti principali
numComponents = 4;
X_four_components = score(:, 1:numComponents);

% Creazione di una nuova figura
figure(6);
set(gcf, 'Position', get(0, 'Screensize'));  % Massimizza la figura

% Creazione della matrice di subplot
for i = 1:numComponents
    for j = 1:numComponents
        subplot(numComponents, numComponents, (i-1)*numComponents + j);
        
        if i == j
            % Istogramma sulla diagonale
            histogram(X_four_components(:,i), 'FaceColor', [0.3 0.3 0.3], 'EdgeColor', 'none');
            title(feature_names{i});
        else
            % Scatter plot fuori dalla diagonale
            scatter(X_four_components(:,j), X_four_components(:,i), 20, target, 'filled');
        end
        
        % Aggiungi etichette solo ai bordi esterni della matrice
        if i == numComponents
            xlabel(feature_names{j});
        end
        if j == 1
            ylabel(feature_names{i});
        end
        
        % Rimuovi i tick label per ridurre il disordine
        set(gca, 'XTickLabel', [], 'YTickLabel', []);
        
        % Aggiungi una griglia leggera
        grid on;
        alpha(0.3);
    end
end

% Aggiungi un titolo generale
sgtitle("Matrix Plot delle Prime Quattro Componenti Principali");

% Crea una legenda corretta
unique_classes = unique(target);
colors = lines(length(unique_classes));  % Genera colori unici per ogni classe

% Crea un nuovo subplot per la legenda
subplot(numComponents, numComponents, numComponents);
hold on;
for i = 1:length(unique_classes)
    scatter([], [], 36, colors(i,:), 'filled', 'DisplayName', sprintf('Classe %d', unique_classes(i)));
end
hold off;

% Nascondi gli assi del subplot della legenda
axis off;

% Aggiungi la legenda
h = legend('show', 'Location', 'eastoutside');
set(h, 'Position', [0.92 0.1 0.07 0.3]);  % Aggiusta la posizione della legenda

% Aggiusta lo spazio tra i subplot
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
set(gcf, 'Units', 'Normalized', 'Position', [0.1 0.1 0.8 0.8]);


%%
% Selezione delle prime quattro componenti principali
numComponents = 4;
X_four_components = score(:, 1:numComponents);

% Creazione di una nuova figura
figure(7);
set(gcf, 'Position', get(0, 'Screensize'));  % Massimizza la figur
% Genera colori unici per ogni classe
unique_classes = unique(target);
colors = lines(length(unique_classes));  % Assicurati di avere colori unici per ogni classe

% Creazione della matrice di subplot
for i = 1:numComponents
    for j = 1:numComponents
        subplot(numComponents, numComponents, (i-1)*numComponents + j);
        
        if i == j
            % Istogramma sulla diagonale
            hold on;  % Mantieni il grafico per aggiungere più histogrammi
            for k = 1:length(unique_classes)
                class_indices = (target == unique_classes(k));  % Trova gli indici della classe k
                
                % Crea l'istogramma per la classe k
                histogram(X_four_components(class_indices, i), ...
                    'FaceColor', colors(k, :), 'EdgeColor', 'none', 'FaceAlpha', 0.5, ...
                    'DisplayName', sprintf('Classe %d', unique_classes(k)));
            end
            hold off;
            title(feature_names{i});
        else
            % Scatter plot fuori dalla diagonale
            hold on;  % Mantieni il grafico per aggiungere più scatter
            for k = 1:length(unique_classes)
                class_indices = (target == unique_classes(k));  % Trova gli indici della classe k
                scatter(X_four_components(class_indices, j), X_four_components(class_indices, i), ...
                    20, colors(k, :), 'filled', 'DisplayName', sprintf('Classe %d', unique_classes(k)));
            end
            hold off;
        end
        
        % Aggiungi etichette solo ai bordi esterni della matrice
        if i == numComponents
            xlabel(feature_names{j});
        end
        if j == 1
            ylabel(feature_names{i});
        end
        
        % Rimuovi i tick label per ridurre il disordine
        set(gca, 'XTickLabel', [], 'YTickLabel', []);
        
        % Aggiungi una griglia leggera
        grid on;
        alpha(0.3);
    end
end

% Aggiungi un titolo generale
sgtitle("Matrix Plot delle Prime Quattro Componenti Principali");

% Crea un nuovo subplot per la legenda
subplot(numComponents, numComponents, numComponents);
hold on;

% Aggiungi le entry della legenda
for k = 1:length(unique_classes)
    scatter([], [], 36, colors(k,:), 'filled', 'DisplayName', sprintf('Classe %d', unique_classes(k)));
end
hold off;

% Nascondi gli assi del subplot della legenda
axis off;

% Aggiungi la legenda
h = legend('show', 'Location', 'eastoutside');
set(h, 'Position', [0.92 0.1 0.07 0.3]);  % Aggiusta la posizione della legenda

% Aggiusta lo spazio tra i subplot
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
set(gcf, 'Units', 'Normalized', 'Position', [0.1 0.1 0.8 0.8]);
