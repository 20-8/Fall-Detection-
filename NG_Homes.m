%% Fall Detection Considering 6 Activities using All available e 2017 Data


clear all
close all
clc
%% Understand how many files we have
Data_dir='S:\radar_group_data\Data sets\Radar Data North Glasgow Homes February 2019\February NG Homes\';

Files = dir(Data_dir);
Files=Files(4:306);
Files = {Files.name};


%% Import Code to Call Particular File with specific number or name




 CC=cell(length(Files), 1);

 
 s1='01R'

for t=1        :length(Files)
    
myPath = Files{1, t}  
C = strsplit(myPath,'A')
CC{t,1}=C{1,2}


tf{t,1} = strncmp(s1,CC{t, 1},3);


end


tf = cell2mat(tf)

Files=Files';
Files = Files(tf,:)


%%
 
% str  = sprintf('%s#', name{:});   
% num  = sscanf(str, '%d.dat#');
% [dummy, index] = sort(num);
% Files = name(index);
tempCounter=0;

z = cell(length(Files), 1);
centroid = cell(length(Files), 1);
bandwidth=cell(length(Files), 1);

 Save_Centroid=cell(length(Files), 1);

for FileIndex=1      :  length(Files)
    tempCounter=tempCounter+1;
    %% Ancortek reading part
    filename=Files(FileIndex);
    fileID = fopen(horzcat(Data_dir,char(filename)), 'r');
    dataArray = textscan(fileID, '%f');
    fclose(fileID);
    
    
    
    radarData = dataArray{1};
    clearvars fileID dataArray ans;
    fc = radarData(1); % Center frequency
    Tsweep = radarData(2); % Sweep time in ms
    Tsweep=Tsweep/1000; %then in sec
    NTS = radarData(3); % Number of time samples per sweep
    Bw = radarData(4); % FMCW Bandwidth. For FSK, it is frequency step;
    % For CW, it is 0.
    Data = radarData(5:end); % raw data in I+j*Q format

    fs=NTS/Tsweep; % sampling frequency ADC
    record_length=length(Data)/NTS*Tsweep; % length of recording in s
    nc=record_length/Tsweep; % number of chirps

%% Reshape data into chirps and do range FFT (1st FFT)
    Data_time=reshape(Data, [NTS nc]);

    % Data_time=Data_time(:,20001:end);

    %Hamming window prior to FFT may reduce the sidelobes in range but maybe not
    %relevant for Doppler (will also need to discard 2 raneg bins not 1 later
    %on!!!)
    % win = repmat(hamming(NTS),1,size(Data_time,2));
    win = ones(NTS,size(Data_time,2));

    
    Data_range=[]; %This helps avoid the mismatch error for files of different length
    
    
    %Part taken from Ancortek code for FFT and IIR filtering
    tmp = fftshift(fft(Data_time.*win),1);
    Data_range(1:NTS/2,:) = tmp(NTS/2+1:NTS,:);
    
    clear tmp %%%%%%%%%%%%%%%%%%%%%%%%%% CHECK THIS
    
    % IIR Notch filter
    ns = oddnumber(size(Data_range,2))-1;
    Data_range_MTI = zeros(size(Data_range,1),ns);
    [b,a] = butter(4, 0.0075, 'high');
    [h, f1] = freqz(b, a, ns);
    
    for k=1:size(Data_range,1)
      Data_range_MTI(k,1:ns) = filter(b,a,Data_range(k,1:ns));
    end
    
    freq =(0:ns-1)*fs/(2*ns); 
    range_axis=(freq*3e8*Tsweep)/(2*Bw);
    %Need to remove the first range bin as it has got some strong residual from
    %filtering?
    Data_range_MTI=Data_range_MTI(2:size(Data_range_MTI,1),:);
    Data_range=Data_range(2:size(Data_range,1),:);

%     figure;
%     colormap(jet)
%     imagesc([1:35000]./1000,range_axis,20*log10(abs(Data_range_MTI)))
%     imagesc(20*log10(abs(Data_range_MTI)))
%     xlabel('No. of Sweeps')
%     ylabel('Range bins')
%     title('Range Profiles after MTI filter')
%     clim = get(gca,'CLim');
%     axis xy; 
%     ylim([1 size(Data_range_MTI,1)])
%     set(gca, 'CLim', clim(2)+[-60,0]);
%     drawnow

%% Spectrogram processing for 2nd FFT to get Doppler
% This selects the range bins where we want to calculate the spectrogram
    bin_indl = 10;
    bin_indu = 60;
    %Parameters for spectrograms
    MD.PRF=1/Tsweep;
    MD.TimeWindowLength = 300;
    MD.OverlapFactor = 0.95;
    MD.OverlapLength = round(MD.TimeWindowLength*MD.OverlapFactor);
    MD.Pad_Factor = 4;
    MD.FFTPoints = MD.Pad_Factor*MD.TimeWindowLength;
    MD.DopplerBin=MD.PRF/(MD.FFTPoints);
    MD.DopplerAxis=-MD.PRF/2:MD.DopplerBin:MD.PRF/2-MD.DopplerBin;
    MD.WholeDuration=size(Data_range_MTI,2)/MD.PRF;
    MD.NumSegments=floor((size(Data_range_MTI,2)-MD.TimeWindowLength)/floor(MD.TimeWindowLength*(1-MD.OverlapFactor)));

    %Method 1 - COHERENT SUM
    % myvec_MTI=sum(Data_range_MTI(bin_indl:bin_indu,:));
    % myvec=sum(Data_range(bin_indl:bin_indu,:));
    % Data_spec_MTI=fftshift(spectrogram(myvec_MTI,MD.TimeWindowLength,MD.OverlapLength,MD.FFTPoints),1);
    % Data_spec=fftshift(spectrogram(myvec,MD.TimeWindowLength,MD.OverlapLength,MD.FFTPoints),1);

    %Method 2 - SUM OF RANGE BINS
    Data_spec_MTI2=0;
    Data_spec2=0;
    
    for RBin=bin_indl:1:bin_indu
        Data_MTI_temp = fftshift(spectrogram(Data_range_MTI(RBin,:),MD.TimeWindowLength,MD.OverlapLength,MD.FFTPoints),1);
        Data_spec_MTI2=Data_spec_MTI2+abs(Data_MTI_temp);                                
        Data_temp = fftshift(spectrogram(Data_range(RBin,:),MD.TimeWindowLength,MD.OverlapLength,MD.FFTPoints),1);
        Data_spec2=Data_spec2+abs(Data_temp);
    end

    %Method 3 - AVERAGE OF RANGE BINS
    % myvec_MTI3=mean(Data_range_MTI(bin_indl:bin_indu,:));
    % myvec3=mean(Data_range(bin_indl:bin_indu,:));
    % Data_spec_MTI3=fftshift(spectrogram(myvec_MTI3,MD.TimeWindowLength,MD.OverlapLength,MD.FFTPoints),1);
    % Data_spec3=fftshift(spectrogram(myvec3,MD.TimeWindowLength,MD.OverlapLength,MD.FFTPoints),1);

    MD.TimeAxis=linspace(0,MD.WholeDuration,size(Data_spec_MTI2,2));

    % Normalise and plot micro-Doppler
    % Data_spec=flipud(Data_spec./max(Data_spec(:)));
    % Data_spec_MTI=flipud(Data_spec_MTI./max(Data_spec_MTI(:)));
    % Data_spec2=flipud(Data_spec2./max(Data_spec2(:)));
%     Data_spec_MTI2=flipud(Data_spec_MTI2./max(Data_spec_MTI2(:)));
    Data_spec_MTI2=flipud(Data_spec_MTI2);
    % Data_spec3=flipud(Data_spec3./max(Data_spec3(:)));
    % Data_spec_MTI3=flipud(Data_spec_MTI3./max(Data_spec_MTI3(:)));

%% Plot Spectrogram here
    Data_contrast=Data_spec_MTI2./(Data_spec_MTI2+mean(Data_spec_MTI2(:))); %Naka-Rushton contrast enhancement
    
    figure
    imagesc(MD.TimeAxis,MD.DopplerAxis,20*log10(abs(Data_spec_MTI2)));
    colormap('jet'); 
    axis xy
    ylim([-500 500]);
    
    colormap; %xlim([1 9])
    clim = get(gca,'CLim');
    set(gca, 'CLim', clim(2)+[-50,0]);
    xlabel('Time[s]', 'FontSize',16);
    ylabel('Doppler [Hz]','FontSize',16)
    set(gca, 'FontSize',16)
    
    axis off
    
%     colorbar
    
%     title(filename)
    
    %% Save the Files in PNG Format
      Data_dir_save = 'C:\Users\ss414j\Desktop\Full Spectrogram\NG Homes all Spectrograms L1 L2 L3\';         %%% Folder Names where Spectrograms are saved
        saveas(gcf,[Data_dir_save,'MD of ',char(filename),'.png'])  % Saves the figure with same name
%     
    %% Doppler and Centroid Extraction
    
    
    MySpect=Data_spec_MTI2;   %%% Spectrograms
    
      DopplerAxis=linspace(-500,500,size(MySpect,1)); 
    
    Doppler_Bin_Cuts=401:800;
    
D1=MySpect(Doppler_Bin_Cuts,:);

DopplerAxis_Cut=DopplerAxis(Doppler_Bin_Cuts);
    
    
     
for timebin=1:floor(size(D1,2)) %%% time bins from 1 to 439 as seen from spectrogram
        num1=0;
        for dopbin=1:floor(size(D1,1)) %For to calculate CENTROID - The Forumula can be found in FF paper (online)
            num1=num1+DopplerAxis_Cut(dopbin).*D1(dopbin,timebin);
        end
        DopC1(1,timebin)=num1./sum(D1(:,timebin));
 end 

centroid{FileIndex,1}=DopC1; %% Save Centroid

for timebin=1:floor(size(D1,2)) %FOR to calculate the bandwidth - The Forumula can be found in FF paper (online)
        numband1=0;
        for dopbin=1:floor(size(D1,1))
            numband1=numband1+(DopplerAxis_Cut(dopbin)-DopC1(1,timebin))^2.*D1(dopbin,timebin);
        end
        BanC1(1,timebin)=sqrt(numband1./sum(D1(:,timebin))) ;                 
end

bandwidth{FileIndex,1}=BanC1; %% Save Bandwidth

%     figure
%     imagesc(MD.TimeAxis,MD.DopplerAxis,20*log10(abs(D1)));
%     colormap('jet'); 
%     set(gca, 'Visible', 'off')
% 
%     axis xy
%     ylim([-500 500]);
%     colorbar
%     colormap; %xlim([1 9])
%     clim = get(gca,'CLim');
%     set(gca, 'CLim', clim(2)+[-40,0]);
%     xlabel('Time[s]', 'FontSize',16);
%     ylabel('Doppler [Hz]','FontSize',16)
%     set(gca, 'FontSize',16)
%     title(filename)
%   
%       Data_dir_save = 'C:\Users\ss414j\Desktop\AllAvailableData\';         %%% Folder Names where Spectrograms are saved
%         saveas(gcf,[Data_dir_save,'MD of ',char(filename),'.png'])  % Saves the figure with same name
%     

%% plot Centroid and Bandwidth here

% figure;
% plot(DopC1)
% hold on
% plot(DopC1+BanC1)
% plot(DopC1-BanC1)
% % %
% % 

Feat2(tempCounter,:) = [mad(BanC1),rms(BanC1),skewness(BanC1),var(BanC1),mean(BanC1), std(BanC1), kurtosis(BanC1),                    mad(DopC1),rms(DopC1),skewness(DopC1),var(DopC1), mean(DopC1), std(DopC1), kurtosis(DopC1)];  %% Three dimentional matrix k is the repetion , m is the total number of movements

 z{FileIndex} = D1;   %

 
 
 
 
 
    figure
%         plot(DopC1+ BanC1);
    hold on
      plot(DopC1);
          

%     plot(DopC1- BanC1);
   
hold off
title(filename)
 
 
  %% Save the Files in PNG Format
%       Data_dir_save = 'C:\Users\ss414j\Desktop\February NG Homes Spectrograms and Centroid\Centroid\';         %%% Folder Names where Spectrograms are saved
%         saveas(gcf,[Data_dir_save,'MD of ',char(filename),'.png'])  % Saves the figure with same name
%  
%         
        Save_Centroid{FileIndex,:}=DopC1;

 clear DopC1
 clear BanC1
 
 
 
 figure
 plot(Save_Centroid{FileIndex,:})
 
 
 [pks,locs] =findpeaks((Save_Centroid{FileIndex,:}),'MinPeakHeight',-5,'MinPeakDistance',200);
hold on 
 x=[1:length(Save_Centroid{FileIndex,:})];
 plot(x(locs),pks,'or')
  title(filename)

 hold off 
 %% Save the Files in PNG Format
      Data_dir_save = 'C:\Users\ss414j\Desktop\February NG Homes Spectrograms and Centroid\Cycles Plot\';         %%% Folder Names where Spectrograms are saved
        saveas(gcf,[Data_dir_save,'MD of ',char(filename),'.png'])  % Saves the figure with same name
    
 
        
        %% Detecting Speed based on maximum and minimum values of centroid
        
        
        a1=max(Save_Centroid{FileIndex,:})  %% Get maximum value of centroid
        b1=min(Save_Centroid{FileIndex,:}) %% Get minimum value of centroid
        
        diff(FileIndex,1)=a1-b1;                 %% calcule diferent of maximum and minimum
        
        %% Calculate the Walking speed of each volunteer from Doppler Frequency formula from radar
        fc=5.8*10^9

speed_light=3*10^8

velocity=(a1*speed_light)/(2*fc)
        
 clear   Save_Centroid{FileIndex,:}
 
   
end %end of loop for all cows files

 



B = reshape(diff,3,[]) ;   %% convert 60 by 1 matrix into 3 x 20 where 3 represent number of repetition for 20 number of people 
 
 
 xax=(17:1:36)
 
 figure
 bar(xax,B')
 xticks([17:1:36])
 
 l{1}='Repetition 1'; l{2}='Repetition 2'; l{3}='Repetition 3';     
legend(l);