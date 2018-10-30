function main()
    [X, T, XTest] = loadData();
    [TTest] = trainDigits(X, T, XTest);
    writePrediction(TTest);
end

function[dataTrain, labelTrain, dataTest] = loadData()
    dataTrainData=load('dataTrain.mat');
    dvars=fields(dataTrainData);
    dataTrain = dataTrainData.(dvars{1});
    
    labelTrainData=load('labelTrain');
    lvars=fields(labelTrainData);
    labelTrain = labelTrainData.(lvars{1});
    
    dataTestData=load('dataTest.mat');
    tvars=fields(dataTestData);
    dataTest = dataTestData.(tvars{1});
end

function [newT] = proccesLabels(T)%transofrma labelurile in vectori de 10 elemente
    
    newT = zeros(10, length(T));
    for id = 1 : length(T)
        class=T(1, id)+1;
        newT(class, id) = 1;
    end
end

function [newT] = unproccesLabels(T)%transofrma vectorii de 10 elemente in labeluri 
    for id = 1 : length(T)
        class = max(T(:, id));
        newT(1, id) = find(T(:, id) == class) - 1;
    end
end

function [TTest] = trainDigits(X, T, XTest)
    [T] = proccesLabels(T);
    %net = fitnet([10]);
    %net = feedforwardnet([10], 'trainlm');
    net = patternnet([30], 'trainlm', 'mse');
    %net.layers{1:2}.transferFcn = 'logsig';
    net.inputs{1}.processFcns = {'fixunknowns', 'mapminmax'};%, 'mapstd'};
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 0.6;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0.  2;
    [trainInd, valInd, testInd] = dividerand(length(X));
    %net.trainParam.max_fail = 10;
    net.trainParam.lr = 0.0001351;
    net.trainParam.epochs = 50000;
    net = configure(net,X,T)
    [net tr] = train(net, X, T);
    view(net);
    
    YTest = sim(net, X(:, testInd));
    [c,cm,ind,per] = confusion(T(:, testInd), YTest)%matricea de confuzie a multimii de test
    plotconfusion(T(:, testInd), YTest)
    
    TTest = sim(net, XTest);
    [ y - sqrt(mse) , y + sqrt(mse) ]
    TTest = unproccesLabels(TTest);
end

function csvwrite_with_headers(filename,m,headers,r,c)%scrie un vector si pune header la coloane intr-un fisier csv(luata de pe net)
    if ~ischar(filename)
        error('FILENAME must be a string');
    end

    % the r and c inputs are optional and need to be filled in if they are
    % missing
    if nargin < 4
        r = 0;
    end
    if nargin < 5
        c = 0;
    end

    if ~iscellstr(headers)
        error('Header must be cell array of strings')
    end


    if length(headers) ~= size(m,2)
        error('number of header entries must match the number of columns in the data')
    end

    % write the header string to the file

    %turn the headers into a single comma seperated string if it is a cell
    %array, 
    header_string = headers{1};
    for i = 2:length(headers)
        header_string = [header_string,',',headers{i}];
    end
    %if the data has an offset shifting it right then blank commas must
    %be inserted to match
    if r>0
        for i=1:r
            header_string = [',',header_string];
        end
end

%write the string to a file
fid = fopen(filename,'w');
fprintf(fid,'%s\r\n',header_string);
fclose(fid);

% write the append the data to the file

%
% Call dlmwrite with a comma as the delimiter
%
dlmwrite(filename, m,'-append','delimiter',',','roffset', r,'coffset',c);
end

function writePrediction(predictedLabels)%scrie solutia intr-un fisier csv
    headers = {'Id', 'Prediction'};
    col1 = 1:12400;
    M = [double(col1)' double(predictedLabels)'];
    csvwrite_with_headers('DumitrescuGabriel.csv', M, headers);
end