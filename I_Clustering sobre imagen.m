%% I. Clustering sobre imagen 


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
