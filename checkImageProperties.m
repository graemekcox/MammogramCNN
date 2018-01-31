clear
close all;

path = '/Users/graemecox/Documents/ResearchProject/Data/Mammograms/';
labels = {'Benign','Cancer','Normal'};

min_x = 10000;
min_y = 10000;
max_x = 0;
max_y = 0;

for i=1:length(labels)
    folder = char(fullfile(path,labels(i)))
    subfolders= dir(folder);
    subfolders(ismember( {subfolders.name}, {'.', '..','.DS_Store'})) = [];
    subfolders.name
    size(subfolders,1)
    size(subfolders,2);
    for j=1:size(subfolders,2)
        new_path = char(fullfile(folder,subfolders(j).name));
        cases= dir(new_path);
        cases(ismember( {cases.name}, {'.', '..','.DS_Store'})) = [];
        cases.name
        for k=1:length(subfolders)
            
            case_path = char(fullfile(new_path,cases(k).name));
            files = dir([case_path,'/*.ics']);
%             cases(ismember( {cases.name}, {'.', '..','.DS_Store'})) = [];
            files.name 
            
            %% Read ICS file
            
            fileID = fopen(fullfile(case_path,char(files.name)),'r');
            line= fgetl(fileID);
            while ischar(line)

                if strfind(line,'LINES')
                    split = strsplit(line,' ');
                    
                    y = str2num(split{3});
                    x = str2num(split{5});
                    
                    % Find min and max x and y values
                    if y > max_y 
                        max_y = y;
                    elseif y< min_y
                        min_y = y;
                    end
                    
                    if x > max_x
                        max_x = x;
                    elseif x < min_x
                        min_x = x;
                    end
                end
%                 if ~isEmpty(strfi??z(split, 'LINES'))
%                     disp('Found it!');
%                     
%                 end
                line = fgetl(fileID);
%                 file_path = char(fullfile(folder,cases(i).name));
%                 files = dir(file_path);
%                 files(ismember( {files.name}, {'.', '..','.DS_Store'})) = [];
%                 files.name  
%             end
            end
        end
    end
end
min_x
min_y
max_x
max_y