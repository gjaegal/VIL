파일을 실행하기 전 아래의 명령어를 실행시켜주세요!
```
mkdir -p NIL/TimeSformer/timesformer/pretrained
wget -O NIL/TimeSformer/timesformer/pretrained/TimeSformer_divST_8x32_224_K400.pyth \
   "https://www.dropbox.com/scl/fi/zcn6byf10i4r0hhojjten/TimeSformer_divST_8x32_224_K400.pyth?rlkey=azfkkmb0qalhgt9vxofhwje54&dl=1"
mv NIL/TimeSformer/timesformer/pretrained/TimeSformer_divST_8x32_224_K400.pyth \
   NIL/TimeSformer/timesformer/pretrained/TimeSformer_divST_8x32_224_K400.pth
```

모델의 용량이 매우 크기 때문에 다운받는데 시간이 오래 걸릴 수 있습니다!

직접 다운 받고 싶으신 분은 아래의 링크에서 다운받아주세요.

[model](https://www.dropbox.com/scl/fi/zcn6byf10i4r0hhojjten/TimeSformer_divST_8x32_224_K400.pyth?rlkey=azfkkmb0qalhgt9vxofhwje54&e=1&dl=0)

그리고 해당 모델을 .pth 형식으로 바꾸어 'NIL/TimeSformer/timesformer/pretrained' 폴더에 저장해주세요.

자세한 내용은 아래의 깃허브를 참고해주시길 바랍니다.

[github](https://github.com/facebookresearch/TimeSformer)
