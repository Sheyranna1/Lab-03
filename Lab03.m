%% Phys Ops Lab 03
% Mason Wahlers, Emily Oeda Rivera, Shey Cajigas 
%% clear command window
clear all; close all; clc;
%% Part 1
% Create an NxN 2-D complex-valued array filled with zeros
N = 64;
%% Part 2
% Define a 2-D bitonal ("black and white") object f[n]
% This is a letter T
object=zeros(N);
object(16:24, 20:44) = 1; 
object(24:42, 28:36) = 1;
%% Part 3
% Assign a random phase to each pixel in the array with a range [-pi  - pi]
% The application of random phase to each pixel has the effect of simulating 
% a diffusely reflecting object because of diff phase at each pixel.
random_phase = (rand(N) - (1/2)) * 2 * pi; 

% The resulting complex-valued array g[n]:
complex_object = object .* exp(1i * random_phase); % This is g[n]
% Complex-valued parts
complex_real = real(complex_object);
complex_img =  imag(complex_object);
complex_mag = abs(complex_object);
complex_phase = angle(complex_object); % Phase in radians
%% Part 4
% Applying the FFT of the 2-D 'centered' complex valued Matrix 
% this produces a complex-valued array of two indices in the 
% "frequency domain," e.g., G[k], where k is the index for the 
% "spatial frequencies" of the sinusoidal components of g[n]. 
fourier_complex = fft2(complex_object); % This is G[k]
% fft of Complex Object Parts
fft_complex_real = real(fourier_complex);
fft_complex_img = imag(fourier_complex);
%% Part 5 
% Compute the magnitude and phase of the 1-D FFT  [] of the array  []:
fft_complex_mag = abs(fourier_complex);
fft_complex_phase = angle(fourier_complex); % Phase in radians
%% Part 6
% Normalize the magnitude at each pixel by dividing by the maximum of all the magnitudes
fft_norm_mag = fft_complex_mag ./ max(fft_complex_mag);
%% Part 7
% Select a cell size for the hologram
% This enlarges your array by factors of 8 in each dimension
%% Part 8
% Quantize the normalized magnitudes so that the largest value is 8
quantized_mag = fft_norm_mag .* 8;
% Then round to the nearest whole number
rounded_quantized_mag = round(quantized_mag);
%% Part 9
% Evaluate Histogram of magnitude which should follow the "Rayleigh" distribution form
figure(1);
% Compute the histogram
[counts, edges] = histcounts(rounded_quantized_mag, 'Normalization', 'pdf');
% Plot the histogram as a line plot
plot(edges(1:end-1), counts, 'LineWidth', 2);
% Labeling and grid
xlabel('Magnitude');
ylabel('Likelihood');
grid on; 
% Title
title('PDF of Magnitude');

%% Part 10
% Quantize the phase at each pixel
quantized_phase = (7 .* fft_complex_phase) / (2*pi) + 4; % Quantize phase to [-4, 3]
% Round to the nearest whole number
rounded_quantized_phase = round(quantized_phase);
% Check Histogram (Should be close to uniformly distributed)
figure(2);
% Compute the histogram
[counts, edges] = histcounts(rounded_quantized_phase, 'Normalization', 'pdf');
% Plot the histogram as a line plot
plot(edges(1:end-1), counts, 'LineWidth', 2);
% Labeling and grid
xlabel('Phase');
ylabel('Likelihood');
grid on; 
% Title
title('PDF of Phase');
%% Part 11
% Expanding matrix and making cells
cell_size = 8;
resized_image = zeros(cell_size * 64);
mag_matrix = rounded_quantized_mag;
phase_matrix = rounded_quantized_phase;

% Loop through each pixel
for i = 1:64% row
    for j = 1:64 % col
        bitonal_apertures = zeros(cell_size);
        mag = mag_matrix(i,j);
        phase = phase_matrix(i,j);

        if mag == 0
            bitonal_apertures(:) = bitonal_apertures;
        else 
            if phase == 0
                bitonal_apertures(9-mag:8,1:2 & 8) = 1;
            elseif phase == 7
                bitonal_apertures(9-mag:8, 7:8 & 1) = 1;
            else
                bitonal_apertures(9-mag:8, phase+1-1:phase+1+1) = 1;
            end
        end   
        resized_image((i-1)*8+1:(i-1)*8+8,(j-1)*8+1:(j-1)*8+8 ) = bitonal_apertures;
    end
end
fft_rendered  =  ifftshift(fft2(fftshift(resized_image)));
fft_rendered_mag = abs(fft_rendered);
fft_log_mag = log(fft_rendered_mag);
norm_fft_log_mag = fft_log_mag ./max(fft_log_mag, [], 'all');

% Display the bitonal apertures
figure(5);
imshow(norm_fft_log_mag);
%% Plotting
figure(3);
subplot(1, 5, 1);
imshow(object, []);
title('Letter T');

subplot(1, 5, 2);
imshow(complex_real, []);
title('Real complex object');

subplot(1, 5, 3);
imshow(complex_img, []);
title('Imaginary complex object');

subplot(1, 5, 4);
imshow(complex_mag, []);
title('Magnitude complex object');

subplot(1, 5, 5);
imshow(complex_phase, []);
title('Phase complex object');

figure(4);
subplot(1, 6, 1);
imshow(object, []);
title('Letter T');

subplot(1, 5, 2);
imshow(fft_complex_real, []);
title('Real part of FFT complex object');

subplot(1, 5, 3);
imshow(fft_complex_img, []);
title('Imaginary part of FFT complex object');

subplot(1, 5, 4);
imshow(fft_complex_mag, []);
title('Magnitude of FFT complex object');

subplot(1, 5, 5);
imshow(fft_complex_phase, []);
title('Phase of FFT complex object');
