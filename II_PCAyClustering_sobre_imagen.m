%% II. PCA y Clustering sobre imagen 

    
 %% Entrega Final: 
% Nombre: Christian Martinez  
% ID: 1067005

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
