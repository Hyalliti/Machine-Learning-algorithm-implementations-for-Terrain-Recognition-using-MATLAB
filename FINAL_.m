 %% Entrega Final: 
% Nombre: Christian Martinez  
% ID: 1067005

% Contenidos:
% 1. Bloque de Imagen// Edicion y Separacion
% 2. Bloque de Clustering // TEMA 1 : PARTE A
% 3. Bloque de Clustering // TEMA 1 : PARTE B
% 4. Bloque de PCA // TEMA 2 : PARTE A
% 5. Bloque de PCA // TEMA 2 : PARTE B
% 6. Bloque de PCA // TEMA 2 : PARTE C
% 7. Bloque de Training Data 
% 8. Bloque de Bayes desicion theory
% 9. Bloque de LDA 
% 10. Bloque de KNN 
% 11. Bloque de Singular Value Decomposition
% 12. OPCIONAL: Arbol de Decisiones
% 13. OPCIONAL: Clasificacion de Zonas por Red Neuronal


% Leyenda: 
% ...**** = Separador de Secciones, Puntos separados tendrán múltiples separadores de secciones.
% Botón de Entrada = Sección de Ejecución que permite borrar variables
% incluidas en la nueva sección de forma voluntaria, permitiendo volver a
% secciones pasadas con carga menor en el Workspace.

% Botón de Pánico = Sección de Ejecución que revierte el código al estado anterior 
.........................................................................**************************************************************


%% % .1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 **************************************************************
%% BLOQUE DE IMAGEN EDICION Y SEPARACION

A=imread('Aerial.jpg');

% RGB BnW           ------ 1.2
subplot(2,2,1)
Rojo = A(:,:,1);
imshow(Rojo);title('Rojo en blanco y negro')

subplot(2,2,2)
Verde = A(:,:,2);
imshow(Verde);title('Verde en blanco y negro')

subplot(2,2,3)
Azul = A(:,:,3);
imshow(Azul);title('Azul en blanco y negro')

% ELIMINACION DE BORDES ---- 1.3
B=double(A);
% para facilidad de los procesos, eliminaremos el borde de 8 pixeles de la
% imágen (donde hay imprecisiones en los cálculos de media y varianzas).
Ax=A(9:end-8,9:end-8,:);

%...........................................................................**************************************************************
% CREACION DE MATRIZ DE REEMPLAZO X --- 1.4
X=zeros(size(Ax,1),size(Ax,2),21,'uint8');
X(:,:,1:3)=Ax;
% DEFINICION DE N'S  ---- 2.0
n1=5;n2=9;n3=17;
n=[n1;n2;n3];
feat_i=4;
% feat_i es el contador del feature en la matriz X.
% N es el tamaño de las áreas de procesamiento.
% CALCULO DE MEDIAS Y VARIANZAS POR REGIóN --- 2.1
for ii=1:3
    h=ones(n(ii))/(n(ii)^2);
    % Calculo medias y varianzas en cada región de tamaño n(ii).
    Mean_i=imfilter(B,h);
    Var_i=imfilter(B.^2,h)-Mean_i.^2;
    
    % recorto las variables resultantes al tamaño efectivo a usar.
    Mean_i=Mean_i(9:end-8,9:end-8,:);
    Var_i=Var_i(9:end-8,9:end-8,:);
    for jj=1:3
        
        % Le resto el mínimo a cada matríz y luego escalo a 255 para poder
        % pasar los valores a uint8 perdiendo el mínimo de información.
        mi=min(min(Mean_i(:,:,ii)));
        Mean_i(:,:,ii)=Mean_i(:,:,ii)-mi;
        ma=max(max(Mean_i(:,:,ii)));
        Mean_i(:,:,ii)= Mean_i(:,:,ii)/ma*255;
        
        mi=min(min(Var_i(:,:,ii)));
        Var_i(:,:,ii)=Var_i(:,:,ii)-mi;
        ma=max(max(Var_i(:,:,ii)));
        Var_i(:,:,ii)=Var_i(:,:,ii)/ma*255;
    end
    
    X(:,:,feat_i:feat_i+2 )   =    uint8(Mean_i);
    X(:,:,feat_i+9:feat_i+11 )=    uint8(Var_i);
    feat_i=feat_i+3;
end
% CREACION DE LA MATRIZ DE TRES COLORES:
% Orden : Z(Colores RGB);Medias RGB , N1N2N3; Varianzas RGB N1N2N3
% Cantidades

% Crear una matriz X,Y,Z
% Donde Z = [R G B] , donde cada valor de color tendra medias n1,n2,n3 y
% ,Varianzas N1,n2,3.

% Imshow Ax muestra el extracto de la original y supongo que es la que hay
% que editar.

% Ax - > Ax [R G B]
% R mas 3 medias de R mas 3 varianzas de R

% Z: II.C.1
color = cat(3,Ax(:,:,1),Ax(:,:,2),Ax(:,:,3));
Rdata = cat(3,X(:,:,4), X(:,:,5),X(:,:,6), X(:,:,13),X(:,:,14) ,X(:,:,15));
Gdata = cat(3,X(:,:,7),X(:,:,8),X(:,:,9),X(:,:,16),X(:,:,17),X(:,:,18));
Bdata = cat(3,X(:,:,10),X(:,:,11),X(:,:,12),X(:,:,19),X(:,:,20),X(:,:,21));
% Z es la matriz de X Reordenada.
Z = cat(3,color,Rdata,Gdata,Bdata);
Z = double(Z(:,:,1:3));
%% % ...........................................................................**************************************************************

%% % ...........................................................................**************************************************************
%% % .2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 **************************************************************
%% BLOQUE DE Clustering :         A  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1    1           1
%% ...........................................................................**************************************************************

%%  PASO 1:  Inicializaciones Aleatorias 
X_Clustering = double(reshape(X,[size(X,1)*size(X,2),21]));
clearvars R_Clustering C1 C2 C3 C4 C5 C6 Clustering

C1 = X_Clustering(randi([1,size(X,1)*size(X,2)]),:); % Cada fila representa un pixel, cada columna representa un feature.
C2 = X_Clustering(randi([1,size(X,1)*size(X,2)]),:);
C3 = X_Clustering(randi([1,size(X,1)*size(X,2)]),:);
C4 = X_Clustering(randi([1,size(X,1)*size(X,2)]),:);
C5 = X_Clustering(randi([1,size(X,1)*size(X,2)]),:);
C6 = X_Clustering(randi([1,size(X,1)*size(X,2)]),:);
Centroides = [C1 ;C2 ;C3 ;C4 ;C5 ;C6]; % Cada fila corresponde a un centroide
%%   PASO 2: Calculo de Distancias

M = 3 ;% Numero de Iteraciones
for M = 1:M
clearvars Calculo_Distancias A i CD1 CD2 CD3 CD4 CD5 CD6
i = 1;
A = 1;
% Preasignacion de La matriz de Distancias
 CD1  = zeros(size(X_Clustering),'like',X_Clustering); 
 CD2  = zeros(size(X_Clustering),'like',X_Clustering); 
 CD3  = zeros(size(X_Clustering),'like',X_Clustering); 
 CD4  = zeros(size(X_Clustering),'like',X_Clustering); 
 CD5  = zeros(size(X_Clustering),'like',X_Clustering); 
 CD6  = zeros(size(X_Clustering),'like',X_Clustering); 

%  Loop de Calculo de Distancias
for A = 1:size(X_Clustering,1)
    CD1(A,:) = (X_Clustering(A,:) - Centroides(1,:));  
    CD2(A,:) = (X_Clustering(A,:) - Centroides(2,:));
    CD3(A,:) = (X_Clustering(A,:) - Centroides(3,:));
    CD4(A,:) = (X_Clustering(A,:) - Centroides(4,:));
    CD5(A,:) = (X_Clustering(A,:) - Centroides(5,:));
    CD6(A,:) = (X_Clustering(A,:) - Centroides(6,:));
end

% Recalculando Centroides
clearvars A
CD1 = sqrt(CD1.^2);CD2 = sqrt(CD2.^2);
CD3 = sqrt(CD3.^2);CD4 = sqrt(CD4.^2);
CD5 = sqrt(CD5.^2);CD6 = sqrt(CD6.^2);

A = 1:size(X_Clustering,1);
C1 = round(mean(CD1 ));C2 = round(mean(CD2 ));
C3 = round(mean(CD3 ));C4 = round(mean(CD4 ));
C5 = round(mean(CD5 ));C6 = round(mean(CD6 ));

Centroides = [C1;C2;C3;C4;C5;C6];
end
%%    PASO 3: Labelling y Asignación de Centroides 
clearvars ClusteringB A 
SCD1 = sum(CD1,2) ;% Suma de las filas formando un vector columna de distancias pixel.
SCD2 = sum(CD2,2) ;
SCD3 = sum(CD3,2) ;
SCD4 = sum(CD4,2) ;
SCD5 = sum(CD5,2) ;
SCD6 = sum(CD6,2) ;
SCD = [SCD1, SCD2, SCD3, SCD4, SCD5, SCD6] ;

[MinSCD,ColumnaSCD] = min(SCD,[],2);
%%     PASO 4: Asignacion de Centroides a Matriz.

clearvars i A ClusteringB
for i =1:length(ColumnaSCD)
    ClusteringBR{i} = Centroides(ColumnaSCD(i),1);
     ClusteringBG{i} = Centroides(ColumnaSCD(i),2);
      ClusteringBB{i} = Centroides(ColumnaSCD(i),3);
end


clearvars A ClusteringA
for A =1:length(ColumnaSCD)
    ClusteringA_R(A,:) = ClusteringBR{A};
    ClusteringA_G(A,:) = ClusteringBG{A};
    ClusteringA_B(A,:) = ClusteringBB{A};
end
%%      PASO 5: Representación de la matriz final 

ClusteringA = cat(2,ClusteringA_R, ClusteringA_G, ClusteringA_B);
R_ClusteringA = uint8(reshape(ClusteringA,size(X,1),size(X,2),3));
figure('Name','Clustering Pixeles Aleatorios')
imshow(R_ClusteringA);
title('Clustering Pixeles Aleatorios')


%% % .3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 **************************************************************
%% BLOQUE DE Clustering :         B  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1    1           1
%% ...........................................................................**************************************************************

%% PASO 6: Incializaciones Manuales 
C1M = X_Clustering(140*311,:); % Calles
C2M = X_Clustering(588*67,:);  % Bosque
C3M =X_Clustering( 287*112,:); % Grama
C4M = X_Clustering(125*129,:); % Viviendas
C5M = X_Clustering(407*7,:); % Edificaciones y parqueos grandes 
C6M =  X_Clustering(330*171,:); % Agua
CentroidesM = [C1M; C2M; C3M; C4M ; C5M; C6M];
%%   PASO 7: Cálculo de Distancias sobre Inicializaciones Manuales
M = 3;% Numero de Iteraciones
for M = 1:M
clearvars Calculo_Distancias A i CD1 CD2 CD3 CD4 CD5 CD6
i = 1;
A = 1;
% Preasignacion de La matriz de Distancias
 CD1  = zeros(size(X_Clustering),'like',X_Clustering); 
 CD2  = zeros(size(X_Clustering),'like',X_Clustering); 
 CD3  = zeros(size(X_Clustering),'like',X_Clustering); 
 CD4  = zeros(size(X_Clustering),'like',X_Clustering); 
 CD5  = zeros(size(X_Clustering),'like',X_Clustering); 
 CD6  = zeros(size(X_Clustering),'like',X_Clustering); 

%  Loop de Calculo de Distancias
for A = 1:size(X_Clustering,1)
    CD1(A,:) = (X_Clustering(A,:) - CentroidesM(1,:));  
    CD2(A,:) = (X_Clustering(A,:) - CentroidesM(2,:));
    CD3(A,:) = (X_Clustering(A,:) - CentroidesM(3,:));
    CD4(A,:) = (X_Clustering(A,:) - CentroidesM(4,:));
    CD5(A,:) = (X_Clustering(A,:) - CentroidesM(5,:));
    CD6(A,:) = (X_Clustering(A,:) - CentroidesM(6,:));
end

% Recalculando CentroidesM
clearvars A
CD1 = sqrt(CD1.^2);CD2 = sqrt(CD2.^2);
CD3 = sqrt(CD3.^2);CD4 = sqrt(CD4.^2);
CD5 = sqrt(CD5.^2);CD6 = sqrt(CD6.^2);

A = 1:size(X_Clustering,1);
clearvars C1 C2 C3 C4 C5 C6 CentroidesM
C1 = round(mean(CD1 ));C2 = round(mean(CD2 ));
C3 = round(mean(CD3 ));C4 = round(mean(CD4 ));
C5 = round(mean(CD5 ));C6 = round(mean(CD6 ));

CentroidesM = [C1;C2;C3;C4;C5;C6];
end
%%     PASO 8:  Asignacion de Labels & Seleccion de Centroides
SCM1 = sum(CD1,2) ;% Suma de las filas formando un vector columna de distancias pixel.
SCM2 = sum(CD2,2) ;
SCM3 = sum(CD3,2) ;
SCM4 = sum(CD4,2) ;
SCM5 = sum(CD5,2) ;
SCM6 = sum(CD6,2) ;

SCM = [SCM1, SCM2, SCM3, SCM4, SCM5, SCM6] ;
Clustering  = zeros(size(X_Clustering),'like',X_Clustering); 
% Asignacion de Centroides 
[MinSCM,ColumnaSCM] = min(SCM,[],2);



clearvars i A ClusteringB
for i =1:length(ColumnaSCD)
    ClusteringBR{i} = Centroides(ColumnaSCD(i),1);
     ClusteringBG{i} = Centroides(ColumnaSCD(i),2);
      ClusteringBB{i} = Centroides(ColumnaSCD(i),3);
end


clearvars A ClusteringA
for A =1:length(ColumnaSCD)
    ClusteringB_R(A,:) = ClusteringBR{A};
    ClusteringB_G(A,:) = ClusteringBG{A};
    ClusteringB_B(A,:) = ClusteringBB{A};
end
 %%      PASO 9: Representación de la matriz final 
ClusteringB = cat(2,ClusteringB_R, ClusteringB_G, ClusteringB_B);
 R_ClusteringManual = uint8(reshape(ClusteringB,size(X,1),size(X,2),3));

figure('Name','Clustering Pixeles Manualmente Elegidos')
 imshow(R_ClusteringManual );
title('Clustering Pixeles Manualmente Elegidos')


%% % ...........................................................................**************************************************************
%% % ...........................................................................**************************************************************

%% %% % .4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 **************************************************************
%% BLOQUE DE PCA Dimensionality Reduction :       A    2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2    2           2
%% ...........................................................................**************************************************************

%% PASO 1: Reshape de la matriz 21 Features para PCA:
X_PCA = double(reshape(X,[size(X,1)*size(X,2),21]));
%%  PASO 2: Creación de Matriz de Transformación y Reducción de Dimensionalidad.
SigmaX = cov(X_PCA);
[EigVa,EigVe] = eig(SigmaX);
EigVa2 = sort(EigVa);EigVa2 = EigVa2(21,:);
% Organiza en una diagonal los valores del Eigen Vector en orden descendente.
W = diag(sort(max(EigVe),'descend'));
% Se elige una reducción de 6 dimensiones
Z_PCA_1 = (W(:,1:end-6)' * X_PCA')';
m = size(Z_PCA_1,2);
% Se realiza Reshape
%%    PASO 3: Transformación de la matriz con Dimensiones reducidas para permitir visualización.
Z_PCA_2 = reshape(Z_PCA_1,[size(X,1),size(X,2),15]);
%%    Reuso de código para cuantización en colores de capas:
imagen_bocachicaR =reshape(Z_PCA_2(:,:,1),[size(X,1)*size(X,2),1] );
imagen_bocachicaG = reshape(Z_PCA_2(:,:,2),[size(X,1)*size(X,2),1]);
imagen_bocachicaB = reshape(Z_PCA_2(:,:,3),[size(X,1)*size(X,2),1]);
% R) Cuantizacion de Capa Roja
clearvars AmpMaxR AmpMinR vectorwowR vectorR  coslsR roswR
AmpMaxR = max(max(imagen_bocachicaR ));
AmpMinR = min(min(imagen_bocachicaR ));
vectorwowR = AmpMinR:(abs(AmpMinR)+abs(AmpMaxR))/255:AmpMaxR;
vectorR = sort(vectorwowR);
[roswR,coslsR] = sort(vectorR);

clear   j bin  imagen_bocachicaR2 i
binR = 1:length(imagen_bocachicaR);
for j = 1:length(imagen_bocachicaR);
    for i = 1:256-1; % Cantidad de niveles de cuantizacion hasta el ultimo
        if vectorR(i) < imagen_bocachicaR(j) & vectorR(i+1) > imagen_bocachicaR(j)
            % Caso superior al minimo
            imagen_bocachicaR2(j) = vectorR(i+1); % Se cuantiza hacia arriba
        elseif vectorR(i+1) == imagen_bocachicaR(j)
            imagen_bocachicaR2(j) = vectorR(i+1);
        end
    end
end

% Transformacion de la imagen en bits

for i = 1:length(imagen_bocachicaR2);
    for j = 1:length(vectorR);
        if imagen_bocachicaR2(i) == vectorR(j);
            ordenR(i) = coslsR(j);
        end
    end
end
ordenR(ordenR == 0) = 1;
ordenR(ordenR == 256) = 255;
% G) Cuantizacion de Capa Verde

AmpMaxG = max(max(imagen_bocachicaG ));
AmpMinG = min(min(imagen_bocachicaG ));
vectorwowG = AmpMinG:(abs(AmpMinG)+abs(AmpMaxG))/255:AmpMaxG;
vectorG = sort(vectorwowG);
[roswG,coslsG] = sort(vectorG);

clear   j bin  imagen_bocachicaG2 i
binG = 1:length(imagen_bocachicaG);
for j = 1:length(imagen_bocachicaG);
    for i = 1:256-1; % Cantidad de niveles de cuantizacion hasta el ultimo
        if vectorG(i) < imagen_bocachicaG(j) & vectorG(i+1) > imagen_bocachicaG(j)
            % Caso superior al minimo
            imagen_bocachicaG2(j) = vectorG(i+1); % Se cuantiza hacia arriba
        elseif vectorG(i+1) == imagen_bocachicaG(j)
            imagen_bocachicaG2(j) = vectorG(i+1);
            
        end
    end
end

% Transformacion de la imagen en bits

for i = 1:length(imagen_bocachicaG2);
    for j = 1:length(vectorG);
        if imagen_bocachicaG2(i) == vectorG(j);
            ordenG(i) = coslsG(j);
        end
    end
end
ordenG(ordenG == 0) = 1;
ordenG(ordenG == 256) = 255;
% B) Cuantizacion de Capa Azul :
AmpMaxB = max(max(imagen_bocachicaB ));
AmpMinB = min(min(imagen_bocachicaB ));
vectorwowB = AmpMinB:(abs(AmpMinB)+abs(AmpMaxB))/255:AmpMaxB;
vectorB = sort(vectorwowB);
[roswB,coslsB] = sort(vectorB);

clear   j bin  imagen_bocachicaB2 i
binB = 1:length(imagen_bocachicaB);
for j = 1:length(imagen_bocachicaB);
    for i = 1:256-1; % Cantidad de niveles de cuantizacion hasta el ultimo
        if vectorB(i) < imagen_bocachicaB(j) & vectorB(i+1) > imagen_bocachicaB(j)
            % Caso superior al minimo
            imagen_bocachicaB2(j) = vectorB(i+1); % Se cuantiza hacia arriba
        elseif vectorB(i+1) == imagen_bocachicaB(j)
            imagen_bocachicaB2(j) = vectorB(i+1);
            
        end
    end
end
% Transformacion de la imagen en bits

for i = 1:length(imagen_bocachicaB2);
    for j = 1:length(vectorB);
        if imagen_bocachicaB2(i) == vectorB(j);
            ordenB(i) = coslsB(j);
        end
    end
end
ordenB(ordenB == 0) = 1;
ordenB(ordenB == 256) = 255;
%  Conversión A Matriz Imagen

PCA_R =uint8(reshape(ordenR,[size(X,1),size(X,2),1]));
PCA_G =uint8(reshape(ordenG,[size(X,1),size(X,2),1]));
PCA_B =uint8(reshape(ordenB,[size(X,1),size(X,2),1]));

Imagen_PCA = cat(3,PCA_R,PCA_G,PCA_B);
%%      PASO 4: Visualización Cuantizada de las Primeras 3 capas.
% Se visualiza el resultado cuantizado que no representa nada en particular.
figure('Name','Visualizacion de PCA, 3 primeras Capas')
imshow(Imagen_PCA);
title('Visualizacion de PCA, 3 primeras Capas')

%% .5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 **************************************************************
%% BLOQUE DE PCA Dimensionality Reduction :       B    2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2    2           2
%% ...........................................................................**************************************************************

%% Boton de entrada a BLOQUE Sección B: 
clearvars ordenR ordenG ordenB SCD SCD1 SCD2 SCD3 SCD4 SCD5 SCD6 CD1 CD2 CD3 CD4 CD5 CD6
%%  Observaciones: 
% Especifcamente en el paso 3 debido a las altas magnitudes de los valores de dimensión Reducida.
% En ambos Bloques, B y C la imagen no puede ser representada como una versión de 255 colores sin alterar la imagen origen incial de Z_PCA_1
%%  PASO 1:  Inicializaciones Aleatorias 

% Se reutiliza el codigo de Clustering utilizando una versión con reshape
% de la Imagen_PCA. 

Z_PCA_B = double(reshape(Imagen_PCA,[size(X,1)*size(X,2),3])) ;

clearvars R_Clustering C1 C2 C3 C4 C5 C6 Clustering

% Z_PCA_1 -> Matriz Unfolded de PCA : Contiene todos los pixeles un una
% sola dimension. Cada Feature es una columna.

% Z_PCA_2,  -> Matriz Acoplada de PCA : Contiene todos los pixeles en sus
% dimensiones respectivas de cada feature.

C1 = Z_PCA_B(randi([1,size(X,1)*size(X,2)]),:); % Cada fila representa un pixel, cada columna representa un feature.
C2 = Z_PCA_B(randi([1,size(X,1)*size(X,2)]),:);
C3 = Z_PCA_B(randi([1,size(X,1)*size(X,2)]),:);
C4 = Z_PCA_B(randi([1,size(X,1)*size(X,2)]),:);
C5 = Z_PCA_B(randi([1,size(X,1)*size(X,2)]),:);
C6 = Z_PCA_B(randi([1,size(X,1)*size(X,2)]),:);
Centroides = [C1 ;C2 ;C3 ;C4 ;C5 ;C6]; % Cada fila corresponde a un centroide
%% DA: 18s   PASO 2: Calculo de Distancias

M = 3 ;% Numero de Iteraciones
for M = 1:M
clearvars Calculo_Distancias A i CD1 CD2 CD3 CD4 CD5 CD6
i = 1;
A = 1;
% Preasignacion de La matriz de Distancias
 CD1  = zeros(size(Z_PCA_B),'like',Z_PCA_B); 
 CD2  = zeros(size(Z_PCA_B),'like',Z_PCA_B); 
 CD3  = zeros(size(Z_PCA_B),'like',Z_PCA_B); 
 CD4  = zeros(size(Z_PCA_B),'like',Z_PCA_B); 
 CD5  = zeros(size(Z_PCA_B),'like',Z_PCA_B); 
 CD6  = zeros(size(Z_PCA_B),'like',Z_PCA_B); 

%  Loop de Calculo de Distancias
for A = 1:size(Z_PCA_B,1)
    CD1(A,:) = (Z_PCA_B(A,:) - Centroides(1,:));  
    CD2(A,:) = (Z_PCA_B(A,:) - Centroides(2,:));
    CD3(A,:) = (Z_PCA_B(A,:) - Centroides(3,:));
    CD4(A,:) = (Z_PCA_B(A,:) - Centroides(4,:));
    CD5(A,:) = (Z_PCA_B(A,:) - Centroides(5,:));
    CD6(A,:) = (Z_PCA_B(A,:) - Centroides(6,:));
end

% Recalculando Centroides
clearvars A
CD1 = sqrt(CD1.^2);CD2 = sqrt(CD2.^2);
CD3 = sqrt(CD3.^2);CD4 = sqrt(CD4.^2);
CD5 = sqrt(CD5.^2);CD6 = sqrt(CD6.^2);

clearvars C1 C2 C3 C4 C5 C6
A = 1:size(Z_PCA_B,1);
C1 = round(mean(CD1 ));C2 = round(mean(CD2 ));
C3 = round(mean(CD3 ));C4 = round(mean(CD4 ));
C5 = round(mean(CD5 ));C6 = round(mean(CD6 ));

Centroides = [C1;C2;C3;C4;C5;C6];
end
%%  DA: 10s PASO 3: Labelling y Asignación de Centroides 
clearvars Clustering_PCA_B A SCD
SCD1 = sum(CD1,2) ;% Suma de las filas formando un vector columna de distancias pixel.
SCD2 = sum(CD2,2) ;
SCD3 = sum(CD3,2) ;
SCD4 = sum(CD4,2) ;
SCD5 = sum(CD5,2) ;
SCD6 = sum(CD6,2) ;
SCD = [SCD1, SCD2, SCD3, SCD4, SCD5, SCD6] ;
% Asignacion de Centroides 


[MinSCD,ColumnaSCD] = min(SCD,[],2);
%% Alternativa 2: 10 segundos
clearvars i A Clustering_PCA_B Clustering_PCA_BR Clustering_PCA_BG Clustering_PCA_BB

for i =1:length(ColumnaSCD)
    ClusteringBR{i} = Centroides(ColumnaSCD(i),1);
     ClusteringBG{i} = Centroides(ColumnaSCD(i),2);
      ClusteringBB{i} = Centroides(ColumnaSCD(i),3);
end


clearvars A ClusteringA
for A =1:length(ColumnaSCD)
    Clustering_PCA_BR(A,:) = ClusteringBR{A};
    Clustering_PCA_BG(A,:) = ClusteringBG{A};
    Clustering_PCA_BB(A,:) = ClusteringBB{A};
end
Clustering_PCA_B = cat(3,Clustering_PCA_BR,Clustering_PCA_BG,Clustering_PCA_BB);
%%     PASO 4: Visualización Cuantizada de las Primeras 3 capas.
Imagen_PCA_B = uint8(reshape(Clustering_PCA_B,[size(X,1),size(X,2),3]));
% Se visualiza el resultado cuantizado que no representa nada en particular.
figure('Name','Visualizacion de PCA, Punto B Clustering Aleatorio, 3 primeras Capas')
imshow(Imagen_PCA_B);
title('Visualizacion de PCA, Punto B Clustering Aleatorio, 3 primeras Capas')

%%  .6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 **************************************************************
%% BLOQUE DE PCA Dimensionality Reduction :       C    2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2    2           2
%% ...........................................................................**************************************************************
%% Boton de entrada a BLOQUE Sección C: 
clearvars ordenR ordenG ordenB SCD SCD1 SCD2 SCD3 SCD4 SCD5 SCD6 CD1 CD2 CD3 CD4 CD5 CD6
%%  PASO 1:  Inicializaciones Aleatorias 

Z_PCA_C = double(reshape(Imagen_PCA,[size(X,1)*size(X,2),3])) ;

clearvars R_Clustering C1 C2 C3 C4 C5 C6 Clustering

C1 = Z_PCA_C(140*311,:); % Calles
C2 =  Z_PCA_C(588*67,:);  % Bosque
C3 = Z_PCA_C( 287*112,:); % Grama
C4 =  Z_PCA_C(125*129,:); % Viviendas
C5 =  Z_PCA_C(407*7,:); % Edificaciones y parqueos grandes 
C6 =   Z_PCA_C(330*171,:); % Agua
Centroides = [C1 ;C2 ;C3 ;C4 ;C5 ;C6]; % Cada fila corresponde a un centroide
%%  DA: 18s PASO 2: Calculo de Distancias

M = 3 ;% Numero de Iteraciones
for M = 1:M
clearvars Calculo_Distancias A i CD1 CD2 CD3 CD4 CD5 CD6
i = 1;
A = 1;
% Preasignacion de La matriz de Distancias
 CD1  = zeros(size(Z_PCA_C),'like',Z_PCA_C); 
 CD2  = zeros(size(Z_PCA_C),'like',Z_PCA_C); 
 CD3  = zeros(size(Z_PCA_C),'like',Z_PCA_C); 
 CD4  = zeros(size(Z_PCA_C),'like',Z_PCA_C); 
 CD5  = zeros(size(Z_PCA_C),'like',Z_PCA_C); 
 CD6  = zeros(size(Z_PCA_C),'like',Z_PCA_C); 

%  Loop de Calculo de Distancias
for A = 1:size(Z_PCA_C,1)
    CD1(A,:) = (Z_PCA_C(A,:) - Centroides(1,:));  
    CD2(A,:) = (Z_PCA_C(A,:) - Centroides(2,:));
    CD3(A,:) = (Z_PCA_C(A,:) - Centroides(3,:));
    CD4(A,:) = (Z_PCA_C(A,:) - Centroides(4,:));
    CD5(A,:) = (Z_PCA_C(A,:) - Centroides(5,:));
    CD6(A,:) = (Z_PCA_C(A,:) - Centroides(6,:));
end

% Recalculando Centroides
clearvars A
CD1 = sqrt(CD1.^2);CD2 = sqrt(CD2.^2);
CD3 = sqrt(CD3.^2);CD4 = sqrt(CD4.^2);
CD5 = sqrt(CD5.^2);CD6 = sqrt(CD6.^2);

clearvars C1 C2 C3 C4 C5 C6
A = 1:size(Z_PCA_C,1);
C1 = round(mean(CD1 ));C2 = round(mean(CD2 ));
C3 = round(mean(CD3 ));C4 = round(mean(CD4 ));
C5 = round(mean(CD5 ));C6 = round(mean(CD6 ));

Centroides = [C1;C2;C3;C4;C5;C6];
end
%%  DA: 10seg  PASO 3: Labelling y Asignación de Centroides 
clearvars Clustering_PCA_C A SCD

SCD1 = sum(CD1,2) ;% Suma de las filas formando un vector columna de distancias pixel.
SCD2 = sum(CD2,2) ;
SCD3 = sum(CD3,2) ;
SCD4 = sum(CD4,2) ;
SCD5 = sum(CD5,2) ;
SCD6 = sum(CD6,2) ;
SCD = [SCD1, SCD2, SCD3, SCD4, SCD5, SCD6] ;
Clustering_C  = zeros(size(Z_PCA_C),'like',Z_PCA_C); 
% Asignacion de Centroides 
[MinSCD,ColumnaSCD] = min(SCD,[],2);

clearvars i A ClusteringB
for i =1:length(ColumnaSCD)
    ClusteringCR{i} = Centroides(ColumnaSCD(i),1);
     ClusteringCG{i} = Centroides(ColumnaSCD(i),2);
      ClusteringCB{i} = Centroides(ColumnaSCD(i),3);
end


clearvars A ClusteringA
for A =1:length(ColumnaSCD)
    Clustering_PCA_CR(A,:) = ClusteringBR{A};
    Clustering_PCA_CG(A,:) = ClusteringBG{A};
    Clustering_PCA_CB(A,:) = ClusteringBB{A};
end
%%     PASO 4: Visualización de las 3 Capas del PCA
Clustering_PCA_C = cat(3,Clustering_PCA_CR,Clustering_PCA_CG,Clustering_PCA_CB);
Imagen_PCA_C = uint8(reshape(Clustering_PCA_C,[size(X,1),size(X,2),3]));
% Se visualiza el resultado cuantizado que no representa nada en particular.
figure('Name','PCA con Clustering  de Pixeles Manualmente Elegidos')
imshow(Imagen_PCA_C);
title('PCA con Clustering  de Manualmente Elegidos')


%% % ...........................................................................**************************************************************
%% % .7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 **************************************************************
%% % .7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 **************************************************************
%% BLOQUE DE TRAINING DATA:           3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3    3           3
%% ...........................................................................**************************************************************

%% PASO 1:  Training Set : Bloques de Coordenadas por Selección Manual de Imagen

% Indizar individualmente cada coordenada
% hacer un for loop que pueda

% Se puede utilizar la matriz AR directamente en vez de operar usando
% columna fila.
% Calle
SeccionC = cell2mat(flip({[232 305];[245 299];[250 295];[254 290];[259 283];[271 276];[274 271];[280 267];[289 262];[298 259];[304 257];[311 256];[321 254];[329 254];[339 251];[346 249];[354 247];[369 243];[374 243];[385 243];[396 242];[407 240];[412 231];[420 226];[428 221];[429 216];[429 206];[429 201];[429 195];[428 190];[427 185];[424 178];[423 169];[423 164];[422 160];[422 156];[422 147];[422 140];[418 113];[419 103];[419 99];[416 78];[416 71];[416 68];[416 62];[413 55];[414 48];[412 39];[412 32];[415 19];[230 307];[209 330];[199 341];[183 354];[169 365];[101 417];[125 400]}));
for i = 1:size(SeccionC,1)
    SeccionC2(i,:) = [SeccionC(i,1)-2 SeccionC(i,2)-2];
    SeccionC3(i,:)= [SeccionC(i,1)-2 SeccionC(i,2)];...
        SeccionC4(i,:) = [SeccionC(i,1)-2 SeccionC(i,2)+2];...
        SeccionC5(i,:)= [SeccionC(i,1) SeccionC(i,2)-2];...
        SeccionC6(i,:)=[SeccionC(i,1) SeccionC(i,2)+2];...
        SeccionC7(i,:)=[SeccionC(i,1)+2 SeccionC(i,2)-2];...
        SeccionC8(i,:)=[SeccionC(i,1)+2 SeccionC(i,2)];...
        SeccionC9(i,:)=[SeccionC(i,1)+2 SeccionC(i,2)+2];
    SeccionC22(i,:) = [SeccionC(i,1)-1 SeccionC(i,2)-1];
    SeccionC33(i,:)= [SeccionC(i,1)-1 SeccionC(i,2)];...
        SeccionC44(i,:) = [SeccionC(i,1)-1 SeccionC(i,2)+1];...
        SeccionC55(i,:)= [SeccionC(i,1) SeccionC(i,2)-1];...
        SeccionC66(i,:)=[SeccionC(i,1) SeccionC(i,2)+1];...
        SeccionC77(i,:)=[SeccionC(i,1)+1 SeccionC(i,2)-1];...
        SeccionC88(i,:)=[SeccionC(i,1)+1 SeccionC(i,2)];...
        SeccionC99(i,:)=[SeccionC(i,1)+1 SeccionC(i,2)+1];
end

Seccion_Callesu = cat(1,SeccionC2, SeccionC22, SeccionC3, SeccionC33, SeccionC4, SeccionC44, SeccionC5, SeccionC55, SeccionC6, SeccionC66, SeccionC7, SeccionC77, SeccionC8, SeccionC88, SeccionC9, SeccionC99);
clearvars SeccionC SeccionC2 SeccionC22 SeccionC3 SeccionC33 SeccionC4 SeccionC44 SeccionC5 SeccionC55 SeccionC6 SeccionC66 SeccionC7 SeccionC77 SeccionC8 SeccionC88 SeccionC9 SeccionC99
% Arboles
SeccionAr = cell2mat(flip({[238 78];[257 83];[205 149];[210 212];[224 206];[400 378];[272 454];[290 450];[226 424];[31 297];[65 29];[47 10];[224 181];[238 182];[346 175];[334 177];[312 174];[290 170];[230 163]}));
for i = 1:size(SeccionAr,1)
    SeccionAr2(i,:) = [SeccionAr(i,1)-2 SeccionAr(i,2)-2];
    SeccionAr3(i,:)= [SeccionAr(i,1)-2 SeccionAr(i,2)];...
        SeccionAr4(i,:) = [SeccionAr(i,1)-2 SeccionAr(i,2)+2];...
        SeccionAr5(i,:)= [SeccionAr(i,1) SeccionAr(i,2)-2];...
        SeccionAr6(i,:)=[SeccionAr(i,1) SeccionAr(i,2)+2];...
        SeccionAr7(i,:)=[SeccionAr(i,1)+2 SeccionAr(i,2)-2];...
        SeccionAr8(i,:)=[SeccionAr(i,1)+2 SeccionAr(i,2)];...
        SeccionAr9(i,:)=[SeccionAr(i,1)+2 SeccionAr(i,2)+2];
    SeccionAr22(i,:) = [SeccionAr(i,1)-1 SeccionAr(i,2)-1];
    SeccionAr33(i,:)= [SeccionAr(i,1)-1 SeccionAr(i,2)];...
        SeccionAr44(i,:) = [SeccionAr(i,1)-1 SeccionAr(i,2)+1];...
        SeccionAr55(i,:)= [SeccionAr(i,1) SeccionAr(i,2)-1];...
        SeccionAr66(i,:)=[SeccionAr(i,1) SeccionAr(i,2)+1];...
        SeccionAr77(i,:)=[SeccionAr(i,1)+1 SeccionAr(i,2)-1];...
        SeccionAr88(i,:)=[SeccionAr(i,1)+1 SeccionAr(i,2)];...
        SeccionAr99(i,:)=[SeccionAr(i,1)+1 SeccionAr(i,2)+1];
end

Seccion_Arbolesu = cat(1,SeccionAr2, SeccionAr22, SeccionAr3, SeccionAr33, SeccionAr4, SeccionAr44, SeccionAr5, SeccionAr55, SeccionAr6, SeccionAr66, SeccionAr7, SeccionAr77, SeccionAr8, SeccionAr88, SeccionAr9, SeccionAr99);
clearvars SeccionAr SeccionAr2 SeccionAr22 SeccionAr3 SeccionAr33 SeccionAr4 SeccionAr44 SeccionAr5 SeccionAr55 SeccionAr6 SeccionAr66 SeccionAr7 SeccionAr77 SeccionAr8 SeccionAr88 SeccionAr9 SeccionAr99
% Grama
SeccionG =  cell2mat(flip({[93 4];[90 11];[83 26];[85 38];[132 36];[311 69];[383 73];[532 148];[504 97];[419 206];[440 228];[491 234];[572 171];[566 232];[673 327];[663 321];[673 317];[383 418];[400 396];[405 413];[386 458];[412 497];[628 490];[678 469];[605 442];[625 448];[631 453];[73 436];[74 469];[185 469];[183 483];[237 443];[9 184];[109 133];[118 128];[167 244];[189 187];[192 197];[192 213];[187 152];[189 136]}));
for i = 1:size(SeccionG,1)
    SeccionG2(i,:) = [SeccionG(i,1)-2 SeccionG(i,2)-2];
    SeccionG3(i,:)= [SeccionG(i,1)-2 SeccionG(i,2)];...
        SeccionG4(i,:) = [SeccionG(i,1)-2 SeccionG(i,2)+2];...
        SeccionG5(i,:)= [SeccionG(i,1) SeccionG(i,2)-2];...
        SeccionG6(i,:)=[SeccionG(i,1) SeccionG(i,2)+2];...
        SeccionG7(i,:)=[SeccionG(i,1)+2 SeccionG(i,2)-2];...
        SeccionG8(i,:)=[SeccionG(i,1)+2 SeccionG(i,2)];...
        SeccionG9(i,:)=[SeccionG(i,1)+2 SeccionG(i,2)+2];
    SeccionG22(i,:) = [SeccionG(i,1)-1 SeccionG(i,2)-1];
    SeccionG33(i,:)= [SeccionG(i,1)-1 SeccionG(i,2)];...
        SeccionG44(i,:) = [SeccionG(i,1)-1 SeccionG(i,2)+1];...
        SeccionG55(i,:)= [SeccionG(i,1) SeccionG(i,2)-1];...
        SeccionG66(i,:)=[SeccionG(i,1) SeccionG(i,2)+1];...
        SeccionG77(i,:)=[SeccionG(i,1)+1 SeccionG(i,2)-1];...
        SeccionG88(i,:)=[SeccionG(i,1)+1 SeccionG(i,2)];...
        SeccionG99(i,:)=[SeccionG(i,1)+1 SeccionG(i,2)+1];
end

Seccion_Gramau = cat(1,SeccionG2, SeccionG22, SeccionG3, SeccionG33, SeccionG4, SeccionG44, SeccionG5, SeccionG55, SeccionG6, SeccionG66, SeccionG7, SeccionG77, SeccionG8, SeccionG88, SeccionG9, SeccionG99);
clearvars SeccionG SeccionG2 SeccionG22 SeccionG3 SeccionG33 SeccionG4 SeccionG44 SeccionG5 SeccionG55 SeccionG6 SeccionG66 SeccionG7 SeccionG77 SeccionG8 SeccionG88 SeccionG9 SeccionG99
% Viviendas/edificios pequeños
SeccionV = cell2mat(flip({[499 284];[536 268];[536 268];[551 280];[564 281];[564 283];[588 322];[597 319];[562 318];[526 299];[536 314];[533 314];[520 311];[520 310];[512 299];[511 298];[501 295];[493 289];[486 289];[485 289];[472 296];[471 296];[469 306];[482 309];[477 307];[475 301];[477 293];[478 293];[498 289];[499 289];[507 300];[511 300];[524 312];[538 273];[537 262];[538 263];[546 273];[531 260];[550 261];[544 271];[536 270];[530 267];[544 257];[556 245];[552 253];[528 266];[540 266];[538 263];[535 245];[531 241];[524 241];[504 248];[504 251];[501 254];[493 259];[499 257];[500 257];[505 253];[509 250];[513 248];[516 254];[516 254];[520 256];[521 256];[522 257];[531 259];[539 260];[556 266];[557 266];[568 281];[568 278];[566 273];[552 275];[551 275];[543 267];[560 270];[568 273];[558 277];[544 248];[555 248];[542 254];[518 250];[518 251];[520 259];[525 251];[527 250];[529 253];[537 300];[503 286];[510 290];[510 290];[516 293];[526 309];[532 298];[258 316];[254 321];[284 296];[320 289];[305 289];[312 294];[308 284];[324 314];[324 314];[336 319];[333 323];[323 323];[327 319];[333 317];[333 317];[322 326];[322 326];[319 322];[318 322];[306 333];[306 333];[304 341];[304 341];[296 340];[290 340];[279 338];[265 328];[265 328];[256 328];[261 321];[260 322];[263 314];[268 310];[272 327];[269 320];[269 320];[278 319];[279 318];[281 316];[275 306];[277 307];[278 308];[281 308];[283 309];[283 307];[285 307];[290 309];[344 292];[346 289];[340 302];[332 309];[334 303];[271 297];[273 298];[277 299];[286 300];[285 302];[287 302];[297 302]}));

for i = 1:size(SeccionV,1)
    SeccionV2(i,:) = [SeccionV(i,1)-2 SeccionV(i,2)-2];
    SeccionV3(i,:)= [SeccionV(i,1)-2 SeccionV(i,2)];...
        SeccionV4(i,:) = [SeccionV(i,1)-2 SeccionV(i,2)+2];...
        SeccionV5(i,:)= [SeccionV(i,1) SeccionV(i,2)-2];...
        SeccionV6(i,:)=[SeccionV(i,1) SeccionV(i,2)+2];...
        SeccionV7(i,:)=[SeccionV(i,1)+2 SeccionV(i,2)-2];...
        SeccionV8(i,:)=[SeccionV(i,1)+2 SeccionV(i,2)];...
        SeccionV9(i,:)=[SeccionV(i,1)+2 SeccionV(i,2)+2];
    SeccionV22(i,:) = [SeccionV(i,1)-1 SeccionV(i,2)-1];
    SeccionV33(i,:)= [SeccionV(i,1)-1 SeccionV(i,2)];...
        SeccionV44(i,:) = [SeccionV(i,1)-1 SeccionV(i,2)+1];...
        SeccionV55(i,:)= [SeccionV(i,1) SeccionV(i,2)-1];...
        SeccionV66(i,:)=[SeccionV(i,1) SeccionV(i,2)+1];...
        SeccionV77(i,:)=[SeccionV(i,1)+1 SeccionV(i,2)-1];...
        SeccionV88(i,:)=[SeccionV(i,1)+1 SeccionV(i,2)];...
        SeccionV99(i,:)=[SeccionV(i,1)+1 SeccionV(i,2)+1];
end

Seccion_Viviendasu = cat(1,SeccionV2, SeccionV22, SeccionV3, SeccionV33, SeccionV4, SeccionV44, SeccionV5, SeccionV55, SeccionV6, SeccionV66, SeccionV7, SeccionV77, SeccionV8, SeccionV88, SeccionV9, SeccionV99);
clearvars SeccionV SeccionV2 SeccionV22 SeccionV3 SeccionV33 SeccionV4 SeccionV44 SeccionV5 SeccionV55 SeccionV6 SeccionV66 SeccionV7 SeccionV77 SeccionV8 SeccionV88 SeccionV9 SeccionV99
% Edificaciones y parqueos grandes
SeccionE = cell2mat(flip({[390 410];[390 408];[393 407];[379 412];[378 411];[381 411];[277 392];[275 390];[275 384];[272 384];[269 381];[272 380];[275 384];[273 384];[270 383];[270 380];[281 379];[283 380];[281 380];[281 378];[281 375];[283 376];[283 378];[282 376];[283 375];[286 374];[288 370];[285 369];[285 369];[287 369];[288 373];[286 374];[283 378];[284 379];[287 380];[289 381];[291 383];[292 382];[297 380];[297 379];[304 376];[305 379];[298 381];[299 379];[301 374];[299 374];[300 374];[302 375];[235 385];[233 384];[233 381];[238 379];[240 383];[236 383];[237 381];[240 379];[243 381];[241 384];[239 382];[240 379];[243 379];[245 381];[242 382];[242 379];[246 380];[245 386];[244 385];[248 381];[253 382];[249 388];[250 385];[253 381];[255 384];[255 388];[252 388];[253 382];[256 382];[258 383];[257 387];[254 385];[258 382];[260 382];[262 385];[259 385];[258 383];[263 384];[274 380];[268 380];[270 380];[273 381];[272 384];[267 384];[267 381];[269 380];[271 381];[271 381];[251 407];[251 407];[250 405];[247 404];[248 406];[250 408];[252 407];[259 421];[258 421];[258 421];[255 418];[254 420];[254 420];[259 424];[255 421];[249 419];[246 419];[244 417];[244 416];[246 416];[248 416];[250 418];[244 477];[243 475];[245 471];[245 472];[245 473];[244 478];[245 479];[246 479];[248 479];[250 478];[250 475];[248 472];[248 471];[251 470];[254 473];[252 479];[252 479];[254 479];[257 478];[257 475];[266 476];[259 482];[262 483];[266 482];[265 479];[272 479];[273 479];[277 482];[279 487];[284 481];[288 478];[660 446];[661 446];[662 448];[665 449];[667 446];[667 445];[668 442];[667 441];[669 441];[666 388];[665 387];[662 386];[660 386];[658 388];[656 390];[654 393];[658 393];[655 393];[651 393];[654 389];[647 397];[644 397];[645 398];[643 398];[638 431];[639 432];[639 433];[642 433];[646 431];[645 428];[638 425];[639 426];[640 427];[645 429];[646 431];[642 430];[639 430];[638 429];[641 429];[616 438];[616 438];[618 437];[613 433];[614 433];[613 438];[615 437];[607 418];[604 415];[605 417];[599 417];[601 418];[604 420];[605 422];[597 420];[599 420];[603 419];[603 418];[607 415];[610 409];[612 408];[613 414];[608 415];[607 415];[607 412];[612 412];[610 409];[613 407];[611 406];[630 378];[631 377];[633 377];[637 377];[636 379];[633 379];[634 377];[639 374];[634 370];[386 214];[385 215];[385 216];[385 217];[384 217];[383 216];[384 214];[386 213];[387 213];[388 214];[390 216];[388 217];[387 216];[388 216];[395 231];[393 231];[393 229];[396 226];[396 227];[394 228];[391 229];[390 231];[392 233];[393 233];[395 233];[397 232];[399 230];[404 226];[400 224];[403 222];[405 222];[399 2];[405 4];[409 3];[550 17];[547 17];[548 22];[548 22];[560 18];[554 19];[443 195];[442 195];[460 186];[463 187];[432 97];[430 106];[435 107];[663 4];[649 4];[639 5];[630 5];[617 5]}));
for i = 1:size(SeccionE,1)
    SeccionE2(i,:) = [SeccionE(i,1)-2 SeccionE(i,2)-2];
    SeccionE3(i,:)= [SeccionE(i,1)-2 SeccionE(i,2)];...
        SeccionE4(i,:) = [SeccionE(i,1)-2 SeccionE(i,2)+2];...
        SeccionE5(i,:)= [SeccionE(i,1) SeccionE(i,2)-2];...
        SeccionE6(i,:)=[SeccionE(i,1) SeccionE(i,2)+2];...
        SeccionE7(i,:)=[SeccionE(i,1)+2 SeccionE(i,2)-2];...
        SeccionE8(i,:)=[SeccionE(i,1)+2 SeccionE(i,2)];...
        SeccionE9(i,:)=[SeccionE(i,1)+2 SeccionE(i,2)+2];
    SeccionE22(i,:) = [SeccionE(i,1)-1 SeccionE(i,2)-1];
    SeccionE33(i,:)= [SeccionE(i,1)-1 SeccionE(i,2)];...
        SeccionE44(i,:) = [SeccionE(i,1)-1 SeccionE(i,2)+1];...
        SeccionE55(i,:)= [SeccionE(i,1) SeccionE(i,2)-1];...
        SeccionE66(i,:)=[SeccionE(i,1) SeccionE(i,2)+1];...
        SeccionE77(i,:)=[SeccionE(i,1)+1 SeccionE(i,2)-1];...
        SeccionE88(i,:)=[SeccionE(i,1)+1 SeccionE(i,2)];...
        SeccionE99(i,:)=[SeccionE(i,1)+1 SeccionE(i,2)+1];
end

Seccion_Edificacionesu = cat(1,SeccionE2, SeccionE22, SeccionE3, SeccionE33, SeccionE4, SeccionE44, SeccionE5, SeccionE55, SeccionE6, SeccionE66, SeccionE7, SeccionE77, SeccionE8, SeccionE88, SeccionE9, SeccionE99);
clearvars SeccionE SeccionE2 SeccionE22 SeccionE3 SeccionE33 SeccionE4 SeccionE44 SeccionE5 SeccionE55 SeccionE6 SeccionE66 SeccionE7 SeccionE77 SeccionE8 SeccionE88 SeccionE9 SeccionE99
% Agua
SeccionA = cell2mat(flip({[233 355];[237 354];[239 348];[240 351];[234 351];[119 105];[113 98];[72 109];[66 109];[716 135];[764 148];[767 192];[669 234];[692 273];[706 287];[706 286];[717 281];[719 335];[728 336];[753 379];[751 399];[747 419];[741 425];[717 437];[736 443];[759 451];[745 495];[737 485];[762 460];[772 432];[774 402];[774 381];[774 370];[774 355];[772 353];[749 350];[723 333];[740 312];[758 307];[774 307];[773 335];[770 327];[759 313];[747 300];[744 294];[737 199];[641 180];[647 187];[671 221];[689 256];[696 277];[723 303];[742 309];[781 306];[777 290];[776 275];[773 258];[769 241];[762 225];[755 217];[730 191];[679 188];[641 177];[647 169];[670 171];[723 175];[775 173];[770 154];[751 146];[747 145];[721 138];[734 119];[753 115];[772 151];[757 151];[699 148];[695 141];[725 151];[761 156];[752 138];[732 138];[700 138];[673 132];[681 124];[711 119];[756 119];[776 102];[760 102];[712 106];[671 105];[674 98];[707 90];[752 89];[778 84];[775 78];[751 78];[727 74];[680 78];[663 88];[645 98];[626 104];[610 114];[634 146];[633 183];[609 178];[596 160];[614 162];[660 165];[688 155];[662 145];[641 138];[617 130];[608 134];[601 150];[596 148];[584 142];[581 134];[580 120];[582 108];[584 90];[584 86];[530 33];[521 41];[506 41];[489 32];[493 31];[504 38];[513 40];[530 37];[529 31];[511 31];[459 18];[469 21];[474 24];[489 27];[496 28];[504 28];[517 33];[553 69];[566 79];[566 69];[594 71];[600 71];[604 71];[615 71];[620 71];[628 72];[638 73];[641 73];[654 73];[668 70];[672 70];[701 76];[716 65];[729 65];[733 66];[738 66];[746 66];[775 64];[774 61];[686 12];[693 12];[702 11];[703 11];[716 10];[728 9];[744 9];[759 9];[759 9];[774 12]}));
for i = 1:size(SeccionA,1)
    SeccionA2(i,:) = [SeccionA(i,1)-2 SeccionA(i,2)-2];
    SeccionA3(i,:)= [SeccionA(i,1)-2 SeccionA(i,2)];...
        SeccionA4(i,:) = [SeccionA(i,1)-2 SeccionA(i,2)+2];...
        SeccionA5(i,:)= [SeccionA(i,1) SeccionA(i,2)-2];...
        SeccionA6(i,:)=[SeccionA(i,1) SeccionA(i,2)+2];...
        SeccionA7(i,:)=[SeccionA(i,1)+2 SeccionA(i,2)-2];...
        SeccionA8(i,:)=[SeccionA(i,1)+2 SeccionA(i,2)];...
        SeccionA9(i,:)=[SeccionA(i,1)+2 SeccionA(i,2)+2];
    SeccionA22(i,:) = [SeccionA(i,1)-1 SeccionA(i,2)-1];
    SeccionA33(i,:)= [SeccionA(i,1)-1 SeccionA(i,2)];...
        SeccionA44(i,:) = [SeccionA(i,1)-1 SeccionA(i,2)+1];...
        SeccionA55(i,:)= [SeccionA(i,1) SeccionA(i,2)-1];...
        SeccionA66(i,:)=[SeccionA(i,1) SeccionA(i,2)+1];...
        SeccionA77(i,:)=[SeccionA(i,1)+1 SeccionA(i,2)-1];...
        SeccionA88(i,:)=[SeccionA(i,1)+1 SeccionA(i,2)];...
        SeccionA99(i,:)=[SeccionA(i,1)+1 SeccionA(i,2)+1];
end

Seccion_Aguau = cat(1,SeccionA2, SeccionA22, SeccionA3, SeccionA33, SeccionA4, SeccionA44, SeccionA5, SeccionA55, SeccionA6, SeccionA66, SeccionA7, SeccionA77, SeccionA8, SeccionA88, SeccionA9, SeccionA99);
clearvars SeccionA SeccionA2 SeccionA22 SeccionA3 SeccionA33 SeccionA4 SeccionA44 SeccionA5 SeccionA55 SeccionA6 SeccionA66 SeccionA7 SeccionA77 SeccionA8 SeccionA88 SeccionA9 SeccionA99
%  PASO 2: Training Set : Bloques de Coordenadas por Separación de Imagen
% El objetivo es Hallar las coordenadas correspondientes a las secciones de
% las imágenes seleccionadas, adjuntadas a la subida y que provienen de la
% imagen original siendo seccionada en 15 partes distintas, luego por
% observacion las imagenes elegidas fueron a su vez divididas en 3
% secciones.

N = round([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]*(size(Z,2)/15));% Vector de Multiplos de Columna de 784(Columnas Totales)/15
% (secciones) Columnas
G = round([1 2 3]*(size(Z,1)/3));% Vector de Multiplos Tercio Superior, Medio e Inferior, .

n = 1:length(N); % Contadores de los Vectores de Multiplos
g = 1:length(G);

% Se puede tomar una matriz de ceros, luego sumarla y elegir las posiciones
% de los valores distintos a cero.
clearvars i j BOT Z1 Z2
% Secciones de Fila
TOP = 1:G(g(1));
MID = G(g(1)):G(g(2));
BOT = G(g(2)):G(g(3));
ROAMER =    1:G(g(3));
% Secciones de Columna
uno =        1:N(1);
dos =         N(1):N(2);
tres=        N(2):N(3);
cuatro=      N(3):N(4);
seis=        N(5):N(6);
ocho=        N(7):N(8);
nueve=       N(8):N(9);
catorce =    N(13):N(14);
quince =    N(14):N(15);

Z1 = zeros([size(Z,1),size(Z,2)]);
% Secciones correspondientes al agua : BOT: 14  ROAMER:15
Z2 = Z1;
for i = TOP
    for j = catorce;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Agua1= Z2+Z1 ;
[RBAgua1,CBAgua1] = find(Z_Bloque_Agua1) ;


% Con este codigo se puede Concatenar todo lo restante y seguir los demás
% normal.
clearvars i j Z2
Z2 = Z1;
for i = ROAMER;
    for j = quince;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Agua2= Z2+Z1 ;
[RBAgua2,CBAgua2] = find(Z_Bloque_Agua2) ;

CoordenadasBloqueAgua = [[RBAgua1,CBAgua1];[RBAgua2,CBAgua2]]; %
% Secciones correspondientes a arboles : TOP: 4
Z2 = Z1;
for i = TOP
    for j = cuatro;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Arboles= Z2+Z1 ;
[RBArboles,CBArboles] = find(Z_Bloque_Arboles) ;
CoordenadasBloqueArboles = [RBArboles,CBArboles];
clearvars i j Z2
% Secciones correspondientes a Calles : MID: 3,4, BOT: 6, TOP : 8,9
Z2 = Z1;
for i = MID
    for j = tres;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Calles1= Z2+Z1 ;
[RBCalles1,CBCalles1] = find(Z_Bloque_Calles1) ;

Z2 = Z1;
for i = MID
    for j = cuatro;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Calles2= Z2+Z1 ;
[RBCalles2,CBCalles2] = find(Z_Bloque_Calles2) ;

Z2 = Z1;
for i = BOT
    for j = seis;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Calles3= Z2+Z1 ;
[RBCalles3,CBCalles3] = find(Z_Bloque_Calles3) ;
Z2 = Z1;
for i = TOP
    for j = ocho;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Calles4= Z2+Z1 ;
[RBCalles4,CBCalles4] = find(Z_Bloque_Calles4) ;
Z2 = Z1;
for i = TOP
    for j = nueve;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Calles5= Z2+Z1 ;
[RBCalles5,CBCalles5] = find(Z_Bloque_Calles5) ;
CoordenadasBloqueCalles = [[RBCalles1,CBCalles1];[RBCalles2,CBCalles2];[RBCalles3,CBCalles3];[RBCalles4,CBCalles4];[RBCalles5,CBCalles5]];
clearvars i j Z2
% Secciones correspondientes a Edificaciones : TOP: 6 , BOT: 8 y 9
Z2 = Z1;
for i = TOP
    for j = seis;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Edificaciones1= Z2+Z1 ;
[RBEdificaciones1,CBEdificaciones1] = find(Z_Bloque_Edificaciones1) ;

clearvars i j Z2
Z2 = Z1;
for i = BOT;
    for j = ocho;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Edificaciones2= Z2+Z1 ;
[RBEdificaciones2,CBEdificaciones2] = find(Z_Bloque_Edificaciones2) ;

clearvars i j Z2
Z2 = Z1;
for i = BOT;
    for j = nueve;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Edificaciones3= Z2+Z1 ;
[RBEdificaciones3,CBEdificaciones3] = find(Z_Bloque_Edificaciones3) ;

CoordenadasBloqueEdificaciones = [[RBEdificaciones1,CBEdificaciones1];[RBEdificaciones2,CBEdificaciones2];[RBEdificaciones3,CBEdificaciones3]]; %
clearvars i j Z2

% Secciones correspondientes a Grama : TOP: 1,2,3,14
Z2 = Z1;
for i = TOP
    for j = uno;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Grama1= Z2+Z1 ;
[RBGrama1,CBGrama1] = find(Z_Bloque_Grama1) ;

clearvars i j Z2
Z2 = Z1;
for i = TOP;
    for j = dos;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Grama2= Z2+Z1 ;
[RBGrama2,CBGrama2] = find(Z_Bloque_Grama2) ;

clearvars i j Z2
Z2 = Z1;
for i = TOP;
    for j = tres;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Grama3= Z2+Z1 ;
[RBGrama3,CBGrama3] = find(Z_Bloque_Grama3) ;
clearvars i j Z2
Z2 = Z1;
for i = BOT;
    for j = catorce;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Grama4= Z2+Z1 ;
[RBGrama4,CBGrama4] = find(Z_Bloque_Grama4) ;

CoordenadasBloqueGrama = [[RBGrama1,CBGrama1];[RBGrama2,CBGrama2];[RBGrama3,CBGrama3];[RBGrama4,CBGrama4]]; %
clearvars i j Z2
% Secciones correspondientes a Viviendas: BOT: 1,2,3,4 MID: 6
Z2 = Z1;
for i = BOT
    for j = uno;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Viviendas1= Z2+Z1 ;
[RBViviendas1,CBViviendas1] = find(Z_Bloque_Viviendas1) ;

Z2 = Z1;
for i = BOT
    for j = dos;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Viviendas2= Z2+Z1 ;
[RBViviendas2,CBViviendas2] = find(Z_Bloque_Viviendas2) ;

Z2 = Z1;
for i = BOT
    for j = tres;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Viviendas3= Z2+Z1 ;
[RBViviendas3,CBViviendas3] = find(Z_Bloque_Viviendas3) ;
Z2 = Z1;
for i = BOT
    for j = cuatro;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Viviendas4= Z2+Z1 ;
[RBViviendas4,CBViviendas4] = find(Z_Bloque_Viviendas4) ;
Z2 = Z1;
for i = MID
    for j = seis;
        Z2(i,j) = 1;
    end
end

Z_Bloque_Viviendas5= Z2+Z1 ;
[RBViviendas5,CBViviendas5] = find(Z_Bloque_Viviendas5) ;
CoordenadasBloqueViviendas = [[RBViviendas1,CBViviendas1];[RBViviendas2,CBViviendas2];[RBViviendas3,CBViviendas3];[RBViviendas4,CBViviendas4];[RBViviendas5,CBViviendas5]];
clearvars i j Z2
%   PASO 3:  Manejo de Coordenadas Iniciales
% 1. Flipar x por y (Usar comando o forloop)
Seccion_Agua1 = Seccion_Aguau(:,2);
Seccion_Agua2 = Seccion_Aguau(:,1);
Seccion_Aguau = cat(2,Seccion_Agua1,Seccion_Agua2);
clearvars Seccion_Agua1 Seccion_Agua2

Seccion_Arboles1 = Seccion_Arbolesu(:,2);
Seccion_Arboles2 = Seccion_Arbolesu(:,1);
Seccion_Arbolesu = cat(2,Seccion_Arboles1,Seccion_Arboles2);
clearvars Seccion_Arboles1 Seccion_Arboles2

Seccion_Calles1 = Seccion_Callesu(:,2);
Seccion_Calles2 = Seccion_Callesu(:,1);
Seccion_Callesu = cat(2,Seccion_Calles1,Seccion_Calles2);
clearvars Seccion_Calles1 Seccion_Calles2

Seccion_Edificaciones1 = Seccion_Edificacionesu(:,2);
Seccion_Edificaciones2 = Seccion_Edificacionesu(:,1);
Seccion_Edificacionesu = cat(2,Seccion_Edificaciones1,Seccion_Edificaciones2);
clearvars Seccion_Edificaciones1 Seccion_Edificaciones2

Seccion_Grama1 = Seccion_Gramau(:,2);
Seccion_Grama2 = Seccion_Gramau(:,1);
Seccion_Gramau = cat(2,Seccion_Grama1,Seccion_Grama2);
clearvars Seccion_Grama1 Seccion_Grama2

Seccion_Viviendas1 = Seccion_Viviendasu(:,2);
Seccion_Viviendas2 = Seccion_Viviendasu(:,1);
Seccion_Viviendasu = cat(2,Seccion_Viviendas1,Seccion_Viviendas2);
clearvars Seccion_Viviendas1 Seccion_Viviendas2

% Fitter, si hay coordenada 0 significa que excede el margen y se puede
% normalizar a 1.
for i = 1:size(Seccion_Edificacionesu,1)
    if Seccion_Edificacionesu(i,1,:)  == 0;
        Seccion_Edificacionesu(i,1,:)  = 1;
    end
end
for i = 1:size(Seccion_Viviendasu,1)
    if Seccion_Viviendasu(i,1,:)  == 0;
        Seccion_Viviendasu(i,1,:)  = 1;
    end
end
for i = 1:size(Seccion_Gramau,1)
    if Seccion_Gramau(i,1,:)  == 0;
        Seccion_Gramau(i,1,:)  = 1;
    end
end
for i = 1:size(Seccion_Callesu,1)
    if Seccion_Callesu(i,1,:)  == 0;
        Seccion_Callesu(i,1,:)  = 1;
    end
end
for i = 1:size(Seccion_Aguau,1)
    if Seccion_Aguau(i,1,:)  == 0;
        Seccion_Aguau(i,1,:)  = 1;
    end
end
for i = 1:size(Seccion_Arbolesu,1)
    
    if Seccion_Arbolesu(i,1,:)  == 0;
        Seccion_Arbolesu(i,1,:)  = 1;
    end
end
%    PASO 4: Limpieza de Variables y Concatenacion de Ambas Selecciones:
Seccion_Agua = [Seccion_Aguau;CoordenadasBloqueAgua];
Seccion_Arbolesu = [ Seccion_Arbolesu;CoordenadasBloqueArboles   ];
Seccion_Callesu= [Seccion_Callesu;CoordenadasBloqueCalles   ];
Seccion_Edificacionesu= [Seccion_Edificacionesu;CoordenadasBloqueEdificaciones   ];
Seccion_Gramau= [Seccion_Gramau;CoordenadasBloqueGrama];
Seccion_Viviendasu= [Seccion_Viviendasu;CoordenadasBloqueViviendas   ];
%     PAS0 5: Hacer una matriz de las mismas dimensiones de Z y
% asignarle un multiplicador 1 a cada una.

Matriz_de_Ceros_Dimension_Z = zeros(size(Z,1),size(Z,2),3);
Y = Matriz_de_Ceros_Dimension_Z;


YAgua = Y;
YArboles = Y;
YCalles = Y;
YEdificacion= Y;
YGrama = Y;
YViviendas = Y;
clearvars i
% Mascara de Agua
for i = 1:size(Seccion_Aguau,1)
    if Z(Seccion_Aguau(i,1,:),Seccion_Aguau(i,2,:),:) ~= 0;
        YAgua(Seccion_Aguau(i,1,:),Seccion_Aguau(i,2,:),:) = 1;
    end
end
% Mascara de Arboles
for i = 1:size(Seccion_Arbolesu,1)
    if Z(Seccion_Arbolesu(i,1,:),Seccion_Arbolesu(i,2,:),:) ~= 0;
        YArboles(Seccion_Arbolesu(i,1,:),Seccion_Arbolesu(i,2,:),:) =1;
    end
end
% Mascara de Grama
for i = 1:size(Seccion_Gramau,1)
    if Z(Seccion_Gramau(i,1,:),Seccion_Gramau(i,2,:),:) ~= 0;
        YGrama(Seccion_Gramau(i,1,:),Seccion_Gramau(i,2,:),:) = 1;
        
    end
end
% Mascara de Calles
for i = 1:size(Seccion_Callesu,1)
    if Z(Seccion_Callesu(i,1,:),Seccion_Callesu(i,2,:),:) ~= 0;
        YCalles(Seccion_Callesu(i,1,:),Seccion_Callesu(i,2,:),:) = 1;
        
    end
end
% Mascara de Viviendas
for i = 1:size(Seccion_Viviendasu,1)
    if Z(Seccion_Viviendasu(i,1,:),Seccion_Viviendasu(i,2,:),:) ~= 0;
        YViviendas(Seccion_Viviendasu(i,1,:),Seccion_Viviendasu(i,2,:),:) =1;
        
    end
end
% Mascara de Edificaciones

Seccion_Edificacionesu = round(Seccion_Edificacionesu,0);
for i = 1:size(Seccion_Edificacionesu,1)
    if Z(Seccion_Edificacionesu(i,1,:),Seccion_Edificacionesu(i,2,:),1:3) ~= 0;
        YEdificacion(Seccion_Edificacionesu(i,1,:),Seccion_Edificacionesu(i,2,:),:) = 1;
    end
end
clearvars i
%%  BOTON DE PANICO
clearvars C_Agua  C_Arboles  C_Calles  C_Edificaciones  C_Grama  C_Viviendas  catorce  CBAgua1  CBAgua2  CBAgua3  CBArboles  CBCalles1  CBCalles2  CBCalles3  CBCalles4  CBCalles5  CBEdificaciones1  CBEdificaciones2  CBEdificaciones3  CBGrama1  CBGrama2  CBGrama3  CBGrama4  CBViviendas1  CBViviendas2  CBViviendas3  CBViviendas4  CBViviendas5  color  CoordenadasBloqueAgua  CoordenadasBloqueArboles  CoordenadasBloqueCalles  CoordenadasBloqueEdificaciones  CoordenadasBloqueGrama  CoordenadasBloqueViviendas  cuatro  dos  feat_i  g  G  Gdata  h  ii  jj  ma  Matriz_de_Ceros_Dimension_Z  Mean_i  mi  MID  n  N  n1  n2  n3  nueve  ocho  quince  RBAgua1  RBAgua2  RBAgua3  RBArboles  RBCalles1  RBCalles2  RBCalles3  RBCalles4  RBCalles5  RBEdificaciones1  RBEdificaciones2  RBEdificaciones3  RBGrama1  RBGrama2  RBGrama3  RBGrama4  RBViviendas1  RBViviendas2  RBViviendas3  RBViviendas4  RBViviendas5  Rdata  ROAMER  Rojo  Seccion_Agua  Seccion_Aguau  Seccion_Arbolesu  Seccion_Callesu  Seccion_Edificacionesu  Seccion_Gramau  Seccion_Viviendasu  seis  TOP  tres  uno  Var_i  Verde  X  Y  YAgua  YArboles  YCalles  YEdificacion  YGrama  YViviendas  Z1  Z_Bloque_Agua1  Z_Bloque_Agua2  Z_Bloque_Agua3  Z_Bloque_Arboles  Z_Bloque_Calles1  Z_Bloque_Calles2  Z_Bloque_Calles3  Z_Bloque_Calles4  Z_Bloque_Calles5  Z_Bloque_Edificaciones1  Z_Bloque_Edificaciones2  Z_Bloque_Edificaciones3  Z_Bloque_Grama1  Z_Bloque_Grama2  Z_Bloque_Grama3  Z_Bloque_Grama4  Z_Bloque_Viviendas1  Z_Bloque_Viviendas2  Z_Bloque_Viviendas3  Z_Bloque_Viviendas4  Z_Bloque_Viviendas5
%%  PAS0 2: Máscaras de Entrenamiento.

% Zona de Agua
C_Agua = uint8(times(YAgua,Z(:,:,1:3)));
figure('Name',' Zonas de Entrenamiento Bloque Agua')
imshow(uint8(C_Agua))
title(' Zonas de Entrenamiento Bloque Agua')
% Zona de Arboles
C_Arboles = uint8(times(YArboles,Z));
figure('Name',' Zonas de Entrenamiento Bloque Arboles')
imshow(uint8(C_Arboles))
title(' Zonas de Entrenamiento Bloque Arboles')
% Zona de Calles
C_Calles = uint8(times(YCalles,Z));
figure('Name',' Zonas de Entrenamiento Bloque Calles')
imshow(uint8(C_Calles))
title(' Zonas de Entrenamiento Bloque Calles')
% Zona de Edificaciones
C_Edificaciones = uint8(times(YEdificacion,Z));
figure('Name',' Zonas de Entrenamiento Bloque Edificaciones')
imshow(uint8(C_Edificaciones))
title(' Zonas de Entrenamiento Bloque Edificaciones')
% Zona de Grama
C_Grama = uint8(times(YGrama,Z));
figure('Name',' Zonas de Entrenamiento Bloque Grama ')
imshow(uint8(C_Grama))
title(' Zonas de Entrenamiento Bloque Grama')
% Zona de Viviendas
C_Viviendas = uint8(times(YViviendas,Z));
figure('Name',' Zonas de Entrenamiento Bloque Viviendas')
imshow(uint8(C_Viviendas))
title(' Zonas de Entrenamiento Bloque Viviendas')
%%   PASO 3: Zona de Entrenamiento Total
Zonas_de_Entrenamiento = C_Viviendas+ C_Agua+ C_Arboles+ C_Calles+ C_Edificaciones+ C_Grama;
figure('Name',' Zonas de Entrenamiento ')
imshow(uint8(Zonas_de_Entrenamiento))
title(' Zonas de Entrenamiento ')
clearvars CBAgua  CBGrama1 Z_Bloque_Agua1  Z_Bloque_Agua2  Z_Bloque_Arboles  Z_Bloque_Calles1  Z_Bloque_Calles2  Z_Bloque_Calles3  Z_Bloque_Calles4  Z_Bloque_Calles5  Z_Bloque_Edificaciones1  Z_Bloque_Edificaciones2  Z_Bloque_Edificaciones3  Z_Bloque_Grama1  Z_Bloque_Grama2  Z_Bloque_Grama3  Z_Bloque_Grama4  Z_Bloque_Viviendas1  Z_Bloque_Viviendas2  Z_Bloque_Viviendas3  Z_Bloque_Viviendas4  Z_Bloque_Viviendas5 RBAgua1  RBAgua2  RBArboles  RBCalles1  RBCalles2  RBCalles3  RBCalles4  RBCalles5  RBEdificaciones1  RBEdificaciones2  RBEdificaciones3  RBGrama1  RBGrama2  RBGrama3  RBGrama4  RBViviendas1  RBViviendas2  RBViviendas3  RBViviendas4  RBViviendas5CBGrama2  CBGrama3  CBGrama4 CBAgua1  CBAgua2  CBArboles  CBCalles1  CBCalles2  CBCalles3  CBCalles4  CBCalles5  CBEdificaciones1  CBEdificaciones2  CBEdificaciones3  CBViviendas1  CBViviendas2  CBViviendas3  CBViviendas4  CBViviendas5

%% ...........................................................................**************************************************************
%%  8   8   8   8   8   8   8   8   8   8   8   8   8   8   8   8   8   8   8   8   8  .**************************************************************
%% BLOQUE DE BAYES DECISION THEORY:           5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5    5           
%% ...........................................................................**************************************************************

%% Creacion de la Matriz Labels aleatorios.
Zonas_de_Entrenamiento_Bayes = Zonas_de_Entrenamiento;
for i = 1:size(Zonas_de_Entrenamiento_Bayes,1)
    for j = 1:size(Zonas_de_Entrenamiento_Bayes,2)
        if Zonas_de_Entrenamiento(i,j,:) == 0
            Zonas_de_Entrenamiento_Bayes(i,j,:) = randi([1 6],1);
        end
    end
end

Zonas_de_Entrenamiento_Bayes1 = double(Zonas_de_Entrenamiento_Bayes);
mcZEN = mean(sum(sum(abs(Zonas_de_Entrenamiento_Bayes1))));
Zonas_de_Entrenamiento_Bayes  = Zonas_de_Entrenamiento_Bayes1/mcZEN; 
Zonas_de_Entrenamiento_Bayes2 = Zonas_de_Entrenamiento_Bayes(:,:,1);
%%  Calculo de probabilidades
class1 =  double(C_Agua);
class2 =  double(C_Arboles); 
class3 =  double(C_Calles); 
class4 =  double(C_Edificaciones);
class5 =  double(C_Grama); 
class6 =  double(C_Viviendas); 

% Se obtienen las  6 clases distintas. Medias preallocadas.
% Luego la probabilidad de que cada valor perteneciente a la variable 1.
mc1 = mean(sum(sum(abs(class1))));
mc2 = mean(sum(sum(abs(class2))));
mc3 = mean(sum(sum(abs(class3))));

mc4 = mean(sum(sum(abs(class4))));
mc5 = mean(sum(sum(abs(class5))));
mc6 = mean(sum(sum(abs(class6))));
% Labels en el plano Coordenado
for i = 1:size(class1,1)
    for j = 1:size(class1,2)
     P1(i,j) = class1(i,j)/mc1;
     P2(i,j) = class2(i,j)/mc2;
     P3(i,j) = class3(i,j)/mc3;
     P4(i,j) = class4(i,j)/mc4;
     P5(i,j) = class5(i,j)/mc5;
     P6(i,j) = class6(i,j)/mc6;
    end
end
% Labels en RGB
P1_Agua =P1;
P2_Arboles = P2;

P3_Calles = P3;
P4_Edificaciones =P4;

P5_Grama = P5;
P6_Viviendas = P6;
%% Creador de Labels
Clases_Labels = cat(3,P1, P2, P3, P4, P5,P6);
for i = 1:size(Clases_Labels,1);
    Zonas_de_Entrenamiento_Bayes(i,j,:) = max(Clases_Labels(i,j,:));
        if Zonas_de_Entrenamiento_Bayes(i,j,:) == P1(i,j)
            LabelClasificado(i,j,:) = 1; % Pw1
        elseif Zonas_de_Entrenamiento_Bayes(i,j,:) == P2(i,j)
            LabelClasificado(i,j,:) = 2; % Pw2
        elseif Zonas_de_Entrenamiento_Bayes(i,j,:) == P3(i,j)
            LabelClasificado(i,j,:) = 3; % Pw3
            
        elseif Zonas_de_Entrenamiento_Bayes(i,j,:) == P4(i,j)
            LabelClasificado(i,j,:) = 4; % Pw3
        elseif Zonas_de_Entrenamiento_Bayes(i,j,:) == P5(i,j)
            LabelClasificado(i,j,:) = 5; % Pw2
        elseif Zonas_de_Entrenamiento_Bayes(i,j,:) == P6(i,j)
            LabelClasificado(i,j,:) = 6; % Pw3
        end
end 
%% Borrador Clasificado Label y Clases
clearvars Clasificado  LabelClasificado Clases_Labels
%% Creacion de Labels para Bayes. Por Secciones.

C2_Agua = double(C_Agua(:,:,:));
for i = 1:size(C_Agua,1)
   for j = 1:size(C_Agua,2)
        if C_Agua(i,j,:)~=0;
            C2_Agua(i,j,:) = 1;
        else 
            C2_Agua(i,j,:) = 0;
        end
    end
end

C2_Arboles = double(C_Arboles(:,:,:));
for i = 1:size(C_Arboles,1)
   for j = 1:size(C_Arboles,2)
        if C_Arboles(i,j,:)~=0;
            C2_Arboles(i,j,:) = 2;
        else 
            C2_Arboles(i,j,:) = 0;
        end
    end
end    


C2_Calles = double(C_Calles(:,:,:));
for i = 1:size(C_Calles,1)
   for j = 1:size(C_Calles,2)
        if C_Calles(i,j,:)~=0;
            C2_Calles(i,j,:) = 3;
        else 
            C2_Calles(i,j,:) = 0;
        end
    end
end      


C2_Edificaciones = double(C_Edificaciones(:,:,:));
for i = 1:size(C_Edificaciones,1)
   for j = 1:size(C_Edificaciones,2)
        if C_Edificaciones(i,j,:)~=0;
            C2_Edificaciones(i,j,:) = 4;
        else 
            C2_Edificaciones(i,j,:) = 0;
        end
    end
end    

C2_Grama = double(C_Grama(:,:,:));
for i = 1:size(C_Grama,1)
   for j = 1:size(C_Grama,2)
        if C_Grama(i,j,:)~=0;
            C2_Grama(i,j,:) = 5;
        else 
            C2_Grama(i,j,:) = 0;
        end
    end
end    

C2_Viviendas = double(C_Viviendas(:,:,:));
for i = 1:size(C_Viviendas,1)
   for j = 1:size(C_Viviendas,2)
        if C_Viviendas(i,j,:)~=0;
            C2_Viviendas(i,j,:) = 6;
        else 
            C2_Viviendas(i,j,:) = 0;
        end
    end
end    





%% ...........................................................................**************************************************************
%%  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10 .**************************************************************
%% BLOQUE DE KNN CLASSIFIER:           6   6   6   6   6   6   6   6   6   6   6   6   6   6   6   6   6   6   6   6    6           6
%% ...........................................................................**************************************************************

%% Predict Classification Using KNN Classifier
Zonas_de_Entrenamiento = double(Zonas_de_Entrenamiento);

% Clases
YAguaKn  =  reshape(YAgua,[size(Zonas_de_Entrenamiento,1)*size(Zonas_de_Entrenamiento,2),3]);;
YArbolesKn  = reshape(YArboles,[size(Zonas_de_Entrenamiento,1)*size(Zonas_de_Entrenamiento,2),3]); ;
YCallesKn  =  reshape(YCalles ,[size(Zonas_de_Entrenamiento,1)*size(Zonas_de_Entrenamiento,2),3]);;
YEdificacionKn  = reshape(YEdificacion,[size(Zonas_de_Entrenamiento,1)*size(Zonas_de_Entrenamiento,2),3]); ;
YGramaKn  = reshape(YGrama,[size(Zonas_de_Entrenamiento,1)*size(Zonas_de_Entrenamiento,2),3]); ;
YViviendasKn  =  reshape(YViviendas,[size(Zonas_de_Entrenamiento,1)*size(Zonas_de_Entrenamiento,2),3]);;
%%  Knn
%%  Aplicando KNN , para K = 1
k = 1
% El Criterio de clasificacion es el color principal al que pertenece
clearvars Distancias IndP1 IndP2 PixelComparado MatrizPixeles MPtransformada Shape1 Shape2 Shape3

% Calculo de distancias
% MatrizPixeles = Zonas_de_Entrenamiento([1:IndP1-1 IndP1+1:end],[1:IndP2-1 IndP2+1:end],:);

MPtransformada = reshape(Zonas_de_Entrenamiento,[size(Zonas_de_Entrenamiento,1)*size(Zonas_de_Entrenamiento,2),3]);
% Las zonas de entrenamiento se transforman a una matriz de 3 columnas
% correspondientes a RGB.
MKnearest = MPtransformada;

clearvars  DminKnearest POSknearest2 i SMPtransformada  SMKnearest POSk1 POS
SMKnearest = sum(MKnearest,2)/10;
SMPtransformada= (sum(MPtransformada,2)/10)';
SMPtransformadaD = SMPtransformada; % Dummy de la SMPtransformada

for i = 1:174    %size(SMPtransformada,1)-1
    SMPtransformada(i) = 0; % El valor correspondiente al del primer pixel será cero.
    [~,POS] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk1(i) = POS(1);
    SMPtransformada(i) = SMPtransformadaD(i) ;
end

PK1 = MKnearest(POSk1,:);
%%  Aplicando KNN , para K = 3

k =3;
clearvars Pixeles_Elegidos POSk1  POSk2  POSk3    DminKnearest POSknearest2 i SMPtransformada SMKnearest POSk1 POS
SMKnearest = sum(MKnearest,2)/10;
SMPtransformada= sum(MPtransformada,2)/10;
SMPtransformadaD = SMPtransformada; % Dummy de la SMPtransformada

for i = 1:1000   %size(SMPtransformada,1)-1
    SMPtransformada(i) = 0; 
    [~,POS] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk1(i) = POS(1);
    SMPtransformada(POS) = 0; 
    
    [~,POS1] = min((SMPtransformada -  SMKnearest(i)).^2); 
    POSk2(i) = POS1(1);
    SMPtransformada(POS1) = 0;
    
    [~,POS2] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk3(i) = POS2(1);
    SMPtransformada= SMPtransformadaD;
end

PK1 = MKnearest(POSk1,:);
PK2 = MKnearest(POSk2,:);
PK3 = MKnearest(POSk3,:);

Pixeles_K3 = [PK1 PK2 PK3];
MPtransformadaPrueba = MPtransformada;
%%  Prueba de Imagen 
MPtransformadaPrueba(1:1000,:) = PK1;
MPtransformadaPrueba2 = reshape(uint8(MPtransformadaPrueba), [size(Zonas_de_Entrenamiento,1),size(Zonas_de_Entrenamiento,2),3]);
%%  Aplicando KNN , para K = 7
clearvars SumatoriaK7 RowK7p1al7 ColK7p1al7 SumatoriaK7 SumatoriaK7Ordenada
k = 7;
k =3;
clearvars Pixeles_Elegidos POSk1  POSk2  POSk3  POSk4  POSk5  POSk6  POSk7 DminKnearest POSknearest2 i SMPtransformada SMKnearest POSk1 POS
SMKnearest = sum(MKnearest,2)/10;
SMPtransformada= sum(MPtransformada,2)/10;
SMPtransformadaD = SMPtransformada; % Dummy de la SMPtransformada

for i = 1:174    %size(SMPtransformada,1)-1
    SMPtransformada(i) = 0; 
    [~,POS1] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk1(i) = POS1(1);
    SMPtransformada(POS1) = 0;
    
    [~,POS2] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk2(i) = POS2(1);
    SMPtransformada(POS2) = 0;
    
    [~,POS3] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk3(i) = POS3(1);
    SMPtransformada(POS3) = 0;
    
    [~,POS4] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk4(i) = POS4(1);
    SMPtransformada(POS4) = 0;
    
    [~,POS5] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk5(i) = POS5(1);
    SMPtransformada(POS5) = 0;
    
    [~,POS6] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk6(i) = POS6(1);
    SMPtransformada(POS6) = 0;
    
    [~,POS7] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk7(i) = POS7(1);
    SMPtransformada(POS7) = 0;
    
    SMPtransformada = SMPtransformadaD;

end

PK1 = MKnearest(POSk1,:);
PK2 = MKnearest(POSk2,:);
PK3 = MKnearest(POSk3,:);
PK4 = MKnearest(POSk4,:);
PK5 = MKnearest(POSk5,:);
PK6 = MKnearest(POSk6,:);
PK7 = MKnearest(POSk7,:);

Pixeles_K7 = [PK1 PK2 PK3 PK4 PK5  PK6 PK7 ];



%% ...........................................................................**************************************************************
%%  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9   9  9  9  .**************************************************************
%% BLOQUE DE LINEAR DISCRIMINANT ANALYSIS:     A)     5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5    5           
%% ...........................................................................**************************************************************

%% A)
%% PASO 1: Matriz de Datos X
% Es la matriz que contiene todos los datos del training set 
% a los cuales se les va a realizar el LDA. 
Zonas_de_Entrenamiento = reshape(Zonas_de_Entrenamiento,[size(C_Agua ,1),size(C_Agua ,2)*size(C_Agua ,3)]) ;
%%  PASO 2: Media de Cada Clase mew sub i.

% Se debe obtener la Media de cada clase, se hace bien si termina en un
% vector de 1x M columnas. 
Clase1 = reshape(C_Agua,[size(C_Agua ,1),size(C_Agua ,2)*size(C_Agua ,3)]) ;
Clase2 = reshape(C_Arboles,[size(C_Agua ,1),size(C_Agua ,2)*size(C_Agua ,3)]) ;
Clase3 = reshape(C_Calles,[size(C_Agua ,1),size(C_Agua ,2)*size(C_Agua ,3)]) ;
Clase4 = reshape(C_Edificaciones,[size(C_Agua ,1),size(C_Agua ,2)*size(C_Agua ,3)]) ;
Clase5 = reshape(C_Grama,[size(C_Agua ,1),size(C_Agua ,2)*size(C_Agua ,3)]) ;
Clase6 = reshape(C_Viviendas,[size(C_Agua ,1),size(C_Agua ,2)*size(C_Agua ,3)]) ;

MClase1 = mean(Clase1);
MClase2 = mean(Clase2);
MClase3 = mean(Clase3);
MClase4 = mean(Clase4);
MClase5 = mean(Clase5);
MClase6 = mean(Clase6);
%%   PASO 3: Media de todas las clases.
TotalClassMeans = mean(Zonas_de_Entrenamiento);
%%    PASO 4: Between Class Matrix.

% Si se utiliza la formula del libro se obtiene un escalar y no una
% matriz.... 
SB1 = ((MClase1-TotalClassMeans)'*(MClase1-TotalClassMeans));
SB2 = ((MClase2-TotalClassMeans)'*(MClase2-TotalClassMeans));
SB3 = ((MClase3-TotalClassMeans)'*(MClase3-TotalClassMeans));
SB4 = ((MClase4-TotalClassMeans)'*(MClase4-TotalClassMeans));
SB5 = ((MClase5-TotalClassMeans)'*(MClase5-TotalClassMeans));
SB6 = ((MClase6-TotalClassMeans)'*(MClase6-TotalClassMeans));
BetweenMatrix_Scatter_Matrix = (SB1+ SB2+ SB3+ SB4+ SB5+ SB6);
SB = BetweenMatrix_Scatter_Matrix;
%%     PASO 5: Within scatter matrix.
 % El resultado debe ser una matriz MxM con 2352 columnas tendrá 2352x2352.
 % La scatter within matriz es una suma de matrices.
 % La within scatter matriz es la suma de las matrices formadas por:
 % Los elementos que componen una clase correspondiente
%  (- menos) 
 % las medias de la clase correspondiente:

% Within Scatter Matrix del Agua. 
for i = 1:size(Clase1,1)
    SW1A(i,:) =  ((Clase1(i,:)' - MClase1'));
    SW1B(i,:) =  ((Clase1(i,:) - MClase1));
end
 SW1 = SW1A'*SW1B ; 
% Within Scatter Matrix de clase Arboles. 
for i = 1:size(Clase2,1)
    SW2A(i,:) =  ((Clase2(i,:)' - MClase2'));
    SW2B(i,:) =  ((Clase2(i,:) - MClase2));
end
 SW2= SW2A'*SW2B ; 
 
% Within Scatter Matrix de Calles. 
for i = 1:size(Clase3,1)
    SW3A(i,:) =  ((Clase3(i,:)' - MClase3'));
    SW3B(i,:) =  ((Clase3(i,:) - MClase3));
end
 SW3= SW3A'*SW3B ; 
 
 % Within Scatter Matrix de Edificaciones. 
for i = 1:size(Clase4,1)
    SW4A(i,:) =  ((Clase4(i,:)' - MClase4'));
    SW4B(i,:) =  ((Clase4(i,:) - MClase4));
end

 SW4= SW4A'*SW4B ; 
% Within Scatter Matrix del Grama. 
for i = 1:size(Clase5,1)
    SW5A(i,:) =  ((Clase5(i,:)' - MClase5'));
    SW5B(i,:) =  ((Clase5(i,:) - MClase5));
end
 SW5= SW5A'*SW5B ; 
% Within Scatter Matrix de Viviendas. 
for i = 1:size(Clase6,1)
    SW6A(i,:) =  ((Clase6(i,:)' - MClase6'));
    SW6B(i,:) =  ((Clase6(i,:) - MClase6));
end
 SW6= SW6A'*SW6B ; 
 
 SW=SW1+SW2+ SW3+ SW4+ SW5+ SW6;
%%      PASO 6: Espacio dimensional reducido
% Se busca tomar el argumento máximo de Wt*SB*W/Wt*Sw*W
% Donde Wt es la matriz de transformación, obtenida como una operación
% entre la Within Scatter Matrix y la Between class Matrix.

%  Debido a las propiedades especiales de la seleccion de elementos se
%  tiene una matriz con una gran cantidad de ceros. Esto causa una matriz
%  Bad Conditioned o mal condicionada, en la que la expresión Ax = B tiende
%  a un modelo lineal Ax = B + b. Si ocurre un pequeño cambio en la matriz,
%  se vería incapaz de adaptarse. Por lo que se convierte en sparse, que
%  son matrices que operan esa cantidad grande de ceros como un valor en
%  específico, reduciendo la cantidad de manera que requiere la operación
%  de borrado a ceros para realizar operaciones, como la inversa de la
%  Within Scatter Matrix. 
% https://la.mathworks.com/help/matlab/ref/rcond.html ;; 
% Para mas informacion referir al PDF adjuntado a la subida.

SPARSE_SW = sparse(SW);
SPARSE_SB = sparse(SB);
SPARSE_W = inv(SPARSE_SW) *SPARSE_SB;

% A esta matriz se le debe de obtener los Eigen Vectores, el criterio
% implica tomar el valor máximo de Eigenvector para describir la
% transformación.
%% Conclusion: Con el training data seleccionado no es posible continuar.
Matriz_Seleccion_PROPIA = ((SPARSE_W')*SPARSE_SB*SPARSE_W)/((SPARSE_W')*SPARSE_SW*SPARSE_W);
%%
[EIGENVECTOR,EIGENVALUE] = eig(Matriz_Seleccion_PROPIA);





%% ...........................................................................**************************************************************
%%  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10  10 .**************************************************************
%% BLOQUE DE KNN CLASSIFIER:           6   6   6   6   6   6   6   6   6   6   6   6   6   6   6   6   6   6   6   6    6           6
%% ...........................................................................**************************************************************

%% Predict Classification Using KNN Classifier
Zonas_de_Entrenamiento = double(Zonas_de_Entrenamiento);

% Clases
YAguaKn  =  reshape(YAgua,[size(Zonas_de_Entrenamiento,1)*size(Zonas_de_Entrenamiento,2),3]);;
YArbolesKn  = reshape(YArboles,[size(Zonas_de_Entrenamiento,1)*size(Zonas_de_Entrenamiento,2),3]); ;
YCallesKn  =  reshape(YCalles ,[size(Zonas_de_Entrenamiento,1)*size(Zonas_de_Entrenamiento,2),3]);;
YEdificacionKn  = reshape(YEdificacion,[size(Zonas_de_Entrenamiento,1)*size(Zonas_de_Entrenamiento,2),3]); ;
YGramaKn  = reshape(YGrama,[size(Zonas_de_Entrenamiento,1)*size(Zonas_de_Entrenamiento,2),3]); ;
YViviendasKn  =  reshape(YViviendas,[size(Zonas_de_Entrenamiento,1)*size(Zonas_de_Entrenamiento,2),3]);;

%% Knn
%%  Aplicando KNN , para K = 1
k = 1
% El Criterio de clasificacion es el color principal al que pertenece
clearvars Distancias IndP1 IndP2 PixelComparado MatrizPixeles MPtransformada Shape1 Shape2 Shape3

% Calculo de distancias
% MatrizPixeles = Zonas_de_Entrenamiento([1:IndP1-1 IndP1+1:end],[1:IndP2-1 IndP2+1:end],:);

MPtransformada = reshape(Zonas_de_Entrenamiento,[size(Zonas_de_Entrenamiento,1)*size(Zonas_de_Entrenamiento,2),3]);
% Las zonas de entrenamiento se transforman a una matriz de 3 columnas
% correspondientes a RGB.
MKnearest = MPtransformada;

clearvars  DminKnearest POSknearest2 i SMPtransformada  SMKnearest POSk1 POS
SMKnearest = sum(MKnearest,2)/10;
SMPtransformada= (sum(MPtransformada,2)/10)';
SMPtransformadaD = SMPtransformada; % Dummy de la SMPtransformada

for i = 1:174    %size(SMPtransformada,1)-1
    SMPtransformada(i) = 0; % El valor correspondiente al del primer pixel será cero.
    [~,POS] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk1(i) = POS(1);
    SMPtransformada(i) = SMPtransformadaD(i) ;
end

PK1 = MKnearest(POSk1,:);
%% Hallar el pixel correspondiente al primer minimo
%%  Aplicando KNN , para K = 3

k =3;
clearvars Pixeles_Elegidos POSk1  POSk2  POSk3    DminKnearest POSknearest2 i SMPtransformada SMKnearest POSk1 POS
SMKnearest = sum(MKnearest,2)/10;
SMPtransformada= sum(MPtransformada,2)/10;
SMPtransformadaD = SMPtransformada; % Dummy de la SMPtransformada

for i = 1:1000   %size(SMPtransformada,1)-1
    SMPtransformada(i) = 0; 
    [~,POS] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk1(i) = POS(1);
    SMPtransformada(POS) = 0; 
    
    [~,POS1] = min((SMPtransformada -  SMKnearest(i)).^2); 
    POSk2(i) = POS1(1);
    SMPtransformada(POS1) = 0;
    
    [~,POS2] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk3(i) = POS2(1);
    SMPtransformada= SMPtransformadaD;
end

PK1 = MKnearest(POSk1,:);
PK2 = MKnearest(POSk2,:);
PK3 = MKnearest(POSk3,:);

Pixeles_K3 = [PK1 PK2 PK3];
MPtransformadaPrueba = MPtransformada;
%% Prueba de Imagen 
MPtransformadaPrueba(1:1000,:) = PK1;
MPtransformadaPrueba2 = reshape(uint8(MPtransformadaPrueba), [size(Zonas_de_Entrenamiento,1),size(Zonas_de_Entrenamiento,2),3]);
%%  Aplicando KNN , para K = 7
clearvars SumatoriaK7 RowK7p1al7 ColK7p1al7 SumatoriaK7 SumatoriaK7Ordenada
k = 7;
k =3;
clearvars Pixeles_Elegidos POSk1  POSk2  POSk3  POSk4  POSk5  POSk6  POSk7 DminKnearest POSknearest2 i SMPtransformada SMKnearest POSk1 POS
SMKnearest = sum(MKnearest,2)/10;
SMPtransformada= sum(MPtransformada,2)/10;
SMPtransformadaD = SMPtransformada; % Dummy de la SMPtransformada

for i = 1:174    %size(SMPtransformada,1)-1
    SMPtransformada(i) = 0; 
    [~,POS1] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk1(i) = POS1(1);
    SMPtransformada(POS1) = 0;
    
    [~,POS2] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk2(i) = POS2(1);
    SMPtransformada(POS2) = 0;
    
    [~,POS3] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk3(i) = POS3(1);
    SMPtransformada(POS3) = 0;
    
    [~,POS4] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk4(i) = POS4(1);
    SMPtransformada(POS4) = 0;
    
    [~,POS5] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk5(i) = POS5(1);
    SMPtransformada(POS5) = 0;
    
    [~,POS6] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk6(i) = POS6(1);
    SMPtransformada(POS6) = 0;
    
    [~,POS7] = min((SMPtransformada -  SMKnearest(i)).^2);
    POSk7(i) = POS7(1);
    SMPtransformada(POS7) = 0;
    
    SMPtransformada = SMPtransformadaD;

end

PK1 = MKnearest(POSk1,:);
PK2 = MKnearest(POSk2,:);
PK3 = MKnearest(POSk3,:);
PK4 = MKnearest(POSk4,:);
PK5 = MKnearest(POSk5,:);
PK6 = MKnearest(POSk6,:);
PK7 = MKnearest(POSk7,:);

Pixeles_K7 = [PK1 PK2 PK3 PK4 PK5  PK6 PK7 ];



%% ...........................................................................**************************************************************
%% 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 ..**************************************************************
%%  SINGULAR VALUE DECOMPOSITION:           7   7   7   7   7   7   7   7   7   7   7   7   7   7   7   7   7   7   7   7    7           7
%% ...........................................................................**************************************************************

%% PASO 1: Aplicar SVD a los datos de la imagen Z:
Z =double(Z);
U = Z(:,:,1:3);
ZR = Z(:,:,1);
ZG = Z(:,:,2);
ZB = Z(:,:,3);

% Los EigenValues están limitados por el numero de filas.
[UR,SR,VR]=svd(ZR); 
[UG,SG,VG]=svd(ZG); 
[UB,SB,VB]=svd(ZB); 

% Se modifica el codigo inicial mostrado en clase
% sabiendo que se aprovecharán todas las
% filas siempre y cuando se determine una cifra M que eventualmente
% representará el porcentaje de elementos de la matriz original que serán
% correspondidos. Como la matriz no es cuadrada el máximo es 515.
%%  PASO 2: Vectores de Valores Propios y Cantidades Totales
Suma_total2R = sum(sum(SR));
Suma_total2G = sum(sum(SG));
Suma_total2B = sum(sum(SB));

% La posicion será el final de el vector incial menos el indice
% correspondiente al elemento. Osea que si el elemento es el primero, su
% posicion será end-1, end-2 al segundo elemento etc.
VER_RED = reshape(SR,1,size(SR,1)*size(SR,2)) ;
VER_RED = unique(VER_RED); 
VER_RED2 = VER_RED;
for i = 1:length(VER_RED2)
    VER_RED(i) = VER_RED2(end+1-i);   
end
VER_RED = VER_RED(1:end-1);

% VERDE
VER_GREEN = reshape(SG,1,size(SG,1)*size(SG,2)) ;
VER_GREEN = unique(VER_GREEN); 
VER_GREEN2 = VER_GREEN;
for i = 1:length(VER_GREEN2)
    VER_GREEN(i) = VER_GREEN2(end+1-i);   
end
VER_GREEN = VER_GREEN(1:end-1);

% AZUL
VER_BLUE = reshape(SB,1,size(SB,1)*size(SB,2)) ;
VER_BLUE = unique(VER_BLUE); 
VER_BLUE2 = VER_BLUE;
for i = 1:length(VER_BLUE2)
    VER_BLUE(i) = VER_BLUE2(end+1-i);   
end
VER_BLUE = VER_BLUE(1:end-1);
%%   PASO 3:Calculo Para Porcentajes
% Porcentaje de 25%
P1 = 25/100; 
for i = 1:length(VER_RED)
    if sum(VER_RED(1:i)) <= round(P1*Suma_total2R) & sum(VER_RED(1:i+1)) > round((P1)*Suma_total2R) 
         j = i;
    else 
        i = i+1;
    end
end
M1R= j  ;clearvars i j;

for i = 1:length(VER_GREEN)
    if sum(VER_GREEN(1:i)) <= round(P1*Suma_total2G) & sum(VER_GREEN(1:i+1)) > round((P1)*Suma_total2G) 
         j = i;
    else 
        i = i+1;
    end
end
M1G = j;clearvars i j;

for i = 1:length(VER_BLUE)
    if sum(VER_BLUE(1:i)) <= round(P1*Suma_total2B) & sum(VER_BLUE(1:i+1)) > round((P1)*Suma_total2B) 
         j = i;
    else 
        i = i+1;
    end
end
M1B = j;clearvars i j;
% Porcentaje 65%
P2 = 65/100;
for i = 1:length(VER_RED)
    if sum(VER_RED(1:i)) <= round(P2*Suma_total2R) & sum(VER_RED(1:i+1)) > round((P2)*Suma_total2R) 
         j = i;
    else 
        i = i+1;
    end
end
M2R = j;clearvars i j;

for i = 1:length(VER_GREEN)
    if sum(VER_GREEN(1:i)) <= round(P2*Suma_total2G) & sum(VER_GREEN(1:i+1)) > round((P2)*Suma_total2G) 
         j = i;
    else 
        i = i+1;
    end
end
M2G = j;clearvars i j;

for i = 1:length(VER_BLUE)
    if sum(VER_BLUE(1:i)) <= round(P2*Suma_total2B) & sum(VER_BLUE(1:i+1)) > round((P2)*Suma_total2B) 
         j = i;
    else 
        i = i+1;
    end
end
M2B = j;clearvars i j;
% Porcentaje 85%
P3 = 85/100;
for i = 1:length(VER_RED)
    if sum(VER_RED(1:i)) <= round(P3*Suma_total2R) & sum(VER_RED(1:i+1)) > round((P3)*Suma_total2R) 
         j = i;
    else 
        i = i+1;
    end
end
M3R = j ;clearvars i j;

for i = 1:length(VER_GREEN)
    if sum(VER_GREEN(1:i)) <= round(P3*Suma_total2G) & sum(VER_GREEN(1:i+1)) > round((P3)*Suma_total2G) 
         j = i;
    else 
        i = i+1;
    end
end
M3G = j;clearvars i j;

for i = 1:length(VER_BLUE)
    if sum(VER_BLUE(1:i)) <= round(P3*Suma_total2B) & sum(VER_BLUE(1:i+1)) > round((P3)*Suma_total2B) 
         j = i;
    else 
        i = i+1;
    end
end
M3B = j;clearvars i j;
%%    PASO 4: Elegir los Single Values que acumulen el 25% de la Suma Total

UhR=UR(:,[1:M1R]);
ShR=SR(1:M1R,1:M1R);
VhR=VR(:,[1:M1R]); 
 
UhG=UG(:,[1:M1G]);
ShG=SG(1:M1G,1:M1G);
VhG=VG(:,[1:M1G]);
 
UhB=UG(:,[1:M1B]);
ShB=SG(1:M1B,1:M1B);
VhB=VG(:,[1:M1B]); 

CR=UhR*ShR*VhR';
CG=UhG*ShG*VhG';
CB=UhB*ShB*VhB';
Crgb1 = cat(3,CR,CG,CB);
figure('Name','Aquellos que acumulan el 25% de la Suma Total')
imshow(uint8(Crgb1))
title('Aquellos que acumulan el 25% de la Suma Total')
clearvars Crgb;
%%     PASO 5: Elegir los Single Values que acumulen el 65% de la Suma Total
UhR=UR(:,[1:M2R]);
ShR=SR(1:M2R,1:M2R);
VhR=VR(:,[1:M2R]); 
 
UhG=UG(:,[1:M2G]);
ShG=SG(1:M2G,1:M2G);
VhG=VG(:,[1:M2G]);
 
UhB=UG(:,[1:M2B]);
ShB=SG(1:M2B,1:M2B);
VhB=VG(:,[1:M2B]); 

CR=UhR*ShR*VhR';
CG=UhG*ShG*VhG';
CB=UhB*ShB*VhB';
Crgb2 = cat(3,CR,CG,CB);
figure('Name','Aquellos que acumulan el 65% de la Suma Total')
imshow(uint8(Crgb2))
title('Aquellos que acumulan el 65% de la Suma Total')
clearvars Crgb;
%%      PASO 6: Elegir los Single Values que acumulen el 85% de la Suma Total
UhR=UR(:,[1:M3R]);
ShR=SR(1:M3R,1:M3R);
VhR=VR(:,[1:M3R]); 
 
UhG=UG(:,[1:M3G]);
ShG=SG(1:M3G,1:M3G);
VhG=VG(:,[1:M3G]);
 
UhB=UG(:,[1:M3B]);
ShB=SG(1:M3B,1:M3B);
VhB=VG(:,[1:M3B]); 

CR=UhR*ShR*VhR';
CG=UhG*ShG*VhG';
CB=UhB*ShB*VhB';
Crgb3 = cat(3,CR,CG,CB);
figure('Name','Aquellos que acumulan el 85% de la Suma Total')
imshow(uint8(Crgb3))
title('Aquellos que acumulan el 85% de la Suma Total')
clearvars Crgb;



