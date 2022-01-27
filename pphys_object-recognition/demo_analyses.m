% This code can be used to do some preliminary analyses with the data
% $ KK 2022

F = 'data/object-recognition/object_recogntion_data_sample.csv';
T = readtable(F,'VariableNamingRule','preserve','NumHeaderLines',0);
%% Get all of the variables from the raw .csv file
images = T.images;
left_choice = T.left_choice;
right_choice = T.right_choice;
reactionTimes = T.rt;
response = T.button_pressed; % 0 = left; 1 = right;
if iscell(response)
    new_response = [];
    for z = 1:length(response)
        if (isempty(response{z})) | (response{z} == 'null')
            new_response = cat(1, new_response, NaN);
        else
           new_response = cat(1, new_response, str2num(response{z}));
        end
    end
    response = new_response;
else
end
%% isolate the valid numbers 
valid_loc = ~isnan(images);
valid_images = images(valid_loc); % image number
valid_left = left_choice(valid_loc); % which object appeared as left choice
valid_right = right_choice(valid_loc); % which object appeared as left choice
valid_response = response(find(valid_loc)+1); % which side did the subject choose
choices = [valid_left valid_right];
valid_rt = str2double(reactionTimes(find(valid_loc)+1)); % reaction time
correct_objNr = repmat(0:9,20,1); correct_objNr = correct_objNr(:); % corresponding correct object number per image
correct_choice = correct_objNr(valid_images+1);
%%  Compile the data table that will be used in future analysis

num_trials = sum(valid_loc); % total trials

% lets fetch all the relevant trial variables for each trial
data = struct;
data.imageNumber = valid_images; % all image numbers
data.target = correct_choice;  % what was the target object number?
for i = 1:num_trials
    data.distractor(i,1) = choices(i,choices(i,:)~= data.target(i));  % what was the distractor object number?
    data.response(i,1) = choices(i,valid_response(i)+1);  % which object with the subject choose?
    data.correct(i,1) = data.response(i,1) == data.target(i,1); % was the subject correct?
end
data.reactionTime = valid_rt; % reaction time of response (in ms)
data = struct2table(data); %% Main data table 

%% plot the object confusion matrix

unique_objects = unique(correct_objNr); % unique object numbers
num_objects = length(unique_objects); % how many objects are there?

obj_confusion_matrix = nan(num_objects,num_objects); % initialize confusion matrix;

for i = 1:num_objects
    for j = 1:num_objects
        if(i~=j)
            obj_confusion_matrix(i,j) = 1 - nanmean(data.correct(data.target == unique_objects(i) & data.distractor == unique_objects(j)));
        end
    end
end
figure;
imagesc(obj_confusion_matrix)
labels = {'bear','elephant','person','car','dog','apple','chair','plane','bird','zebra'};
set(gca,'xtick', 1:10,...
    'xticklabel',labels,...
    'ytick', 1:10,...
    'yticklabel',labels,...
    'TickDir', 'out',...
    'ticklength',[0.019 0.019],...
    'box', 'off' ,...
    'FontName','Arial',...
    'FontSize',15,...
    'PlotBoxAspectRatio',[1 1 1],...
    'DefaultAxesFontName', 'Arial');
xtickangle(45)
xlabel('Distractors')
ylabel('Target')
title('Object Confusion Matrix')
colorbar