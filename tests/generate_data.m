% tests/generate_data.py
% Utility script to generate test data using the Matlab implementation [2]
% Copyright (C) 2020 Eirik Ekjord Vesterkjaer
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
%
% Author:
%     Eirik Ekjord Vesterkjaer
%     eve.eirik@gmail.com
%     github.com/eirikeve
%
% Sources:
%     [2] Jia, Bin: Sparse Gauss-Hermite Quadrature rule
%         https://github.com/binjiaqm/sparse-Gauss-Hermite-quadrature-rule
%         GitHub repository, Matlab code (commit 4afe0bc)
%
% Requirements:
%     Original SGHQ Matlab codes from Bin Jia [2]
%
% Usage:
%     Place in same directory as the .m files from [2]
%     Run this file using Matlab or Octave
%     CSVs will be written under data/
%     Copy the data/ folder you got to tests/ folder in the python-sghq repo
%
% See README.md for more information, and LICENSE for the license.

Manner=3 % Accuracy L -> Point of dim 2^L-1, see get_1d_point.m in [2]

if ~exist('data', 'dir')
    disp('Making data/ directory')
    mkdir('data');
end

for L = 1:5
    for n = 1:6
        [X, W] = generate_md_points(L,n,Manner);
        filename_X = strcat( 'data/SGHQ-X-L-', num2str(L), '-n-', num2str(n), '.csv');
        filename_W = strcat( 'data/SGHQ-W-L-', num2str(L), '-n-', num2str(n), '.csv');
        disp(strcat('Writing: ', filename_X))
        csvwrite(filename_X, X')
        disp(strcat('Writing: ', filename_W))
        csvwrite(filename_W, W')
    end
end

disp('Finished')
