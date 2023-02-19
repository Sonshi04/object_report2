%リランキングなしの正解率39/100
n=0; list={};
LIST={'pos','bgimg'};
DIR0="./";
for i=1:length(LIST)
    DIR=strcat(DIR0,LIST(i),'/');
    W=dir(DIR{:});
    for j=1:size(W)
      if (strfind(W(j).name,'.jpg'))
        fn=strcat(DIR{:},W(j).name);
        n=n+1;
        fprintf('[%d] %s\n',n,fn);
        list={list{:} fn};
      end
    end
end
list_interesting={};
LIST_intersting={'pos_interesting'};
DIR0="./";
for i=1:length(LIST_intersting)
    DIR=strcat(DIR0,LIST_intersting(i),'/');
    W=dir(DIR{:});
    for j=1:size(W)
      if (strfind(W(j).name,'.jpg'))
        fn=strcat(DIR{:},W(j).name);
        n=n+1;
        fprintf('[%d] %s\n',n,fn);
        list_interesting={list_interesting{:} fn};
      end
    end
end
%ポジティブ,ネガティブ画像の使用枚数
n_pos = 25;
n_neg = 500;
PosList=list(1:n_pos); 
NegList=list(n_pos+1:n_pos+n_neg); 
Training={PosList{:} NegList{:}};

net = alexnet;
IM = [];
for i=1:n_pos
    img = imread(PosList{i});
    reimg = imresize(img,net.Layers(1).InputSize(1:2)); 
    IM = cat(4,IM,reimg);
end
for i=1:n_neg
    img = imread(NegList{i});
    reimg = imresize(img,net.Layers(1).InputSize(1:2)); 
    IM = cat(4,IM,reimg);
end
%relevanceで学習
dcnnf = activations(net,IM,'fc7');
dcnnf = squeeze(dcnnf);
dcnnf = (dcnnf/norm(dcnnf))';
training_label = [ones(n_pos,1); ones(n_neg,1)*(-1)];
model_lin = fitcsvm(dcnnf,training_label,'KernelFunction','linear','KernelScale','auto');
%interestingのdcnnf作成
IM_interesting = [];
%301はグレースケールを1枚含むため300枚にするための数合わせ
for i=1:300
    img = imread(list_interesting{i});
    %グレースケール排除
    if ndims(img) ~= 3
        continue
    end
    reimg = imresize(img,net.Layers(1).InputSize(1:2)); 
    IM_interesting = cat(4,IM_interesting,reimg);
end
dcnnf_interesting = activations(net,IM_interesting,'fc7');  
dcnnf_interesting = squeeze(dcnnf_interesting);
dcnnf_interesting = (dcnnf_interesting/norm(dcnnf_interesting))';
%interestingでテスト
[predicted_label,scores] = predict(model_lin, dcnnf_interesting);
[sorted_score,sorted_idx] = sort(scores(:,2),'descend');
FID = fopen('exp.txt','w');
for i=1:numel(sorted_idx)
  fprintf(FID,'%s %f\n',list_interesting{sorted_idx(i)},sorted_score(i));
end
fclose(FID);