library(latex2exp)
#library(scales)
library(ggplot2)

BASE_DIR = "~/sshfs/code/early-sexual-predator-detection-lab/"

configs = list(

  #################### BERT-large

  c(
    "2020-12-27_04-09-08__bert-large-uncased_on_PANC_with_seq-len-512",
    "non_quantized",
    "BERT-large on PANC"
  )
  , c(
    "2020-12-27_11-56-28__bert-large-uncased_on_PANC_with_seq-len-512", # best, s=5
    "non_quantized",
    "BERT-large on PANC"
  )
  , c(
    "2020-12-27_15-59-58__bert-large-uncased_on_PANC_with_seq-len-512",
    "non_quantized",
    "BERT-large on PANC"
  )
  #
  # ##################### BERT-base
  #
  , c(
    "2020-11-21_23-11-03__bert_classifier_on_PANC_with_seq-len-512",
    "non_quantized",
    "BERT-base on PANC"
  )
  # c(
  #   "2020-11-22_00-45-54__bert_classifier_on_PANC_with_seq-len-512",
  #   "non_quantized",
  #   "BERT-base on PANC"
  # ),
  # c(
  #   "2020-11-30_15-06-54__bert_classifier_on_PANC_with_seq-len-512",
  #   "non_quantized",
  #   "BERT-base on PANC"
  # ),
  #
  # # #################### MobileBERT
  #
  # c(
  #   "2020-11-30_15-59-17__mobilebert_classifier_on_PANC_with_seq-len-512",
  #   "non_quantized",
  #   "MobileBERT on PANC"
  # ),
  , c(
    "2020-11-30_17-08-16__mobilebert_classifier_on_PANC_with_seq-len-512",
    "non_quantized",
    "MobileBERT on PANC"
  )
  , c(
    "2020-11-30_17-32-51__mobilebert_classifier_on_PANC_with_seq-len-512",
    "non_quantized",
    "MobileBERT on PANC"
  )
)

savePDF = TRUE


for (config in configs) {

  run=config[1]
  model_version=config[2]
  title=config[3]
  pdf_out_file = paste(sep="", BASE_DIR, "r_scripts/plots/mean_multi_f1_over_percentage_of_information.pdf")

  savePath=paste(sep="","~/sshfs/code/early-sexual-predator-detection-lab/resources/",run,"/",model_version,"/message_based_eval/metrics by skepticism.pdf")
  print(run)
  print(title)

  f1_dat = readLines(paste(sep="", "~/sshfs/code/early-sexual-predator-detection-lab/resources/",run,"/",model_version,"/message_based_eval/metrics_by_skepticism/f1.txt"))
  f1 = as.numeric(f1_dat)

  precision_dat = readLines(paste(sep="", "~/sshfs/code/early-sexual-predator-detection-lab/resources/",run,"/",model_version,"/message_based_eval/metrics_by_skepticism/precision.txt"))
  precision = as.numeric(precision_dat)

  recall_dat = readLines(paste(sep="", "~/sshfs/code/early-sexual-predator-detection-lab/resources/",run,"/",model_version,"/message_based_eval/metrics_by_skepticism/recall.txt"))
  recall = as.numeric(recall_dat)

  speed_dat = readLines(paste(sep="", "~/sshfs/code/early-sexual-predator-detection-lab/resources/",run,"/",model_version,"/message_based_eval/metrics_by_skepticism/speed.txt"))
  speed = as.numeric(speed_dat)

  f_latency_dat = readLines(paste(sep="", "~/sshfs/code/early-sexual-predator-detection-lab/resources/",run,"/",model_version,"/message_based_eval/metrics_by_skepticism/f_latency.txt"))
  f_latency = as.numeric(f_latency_dat)


  par(mfrow=c(1,1))
  par(mar=c(5, 4, 4, 6) + 0.1)

  main = paste(sep="","Metrics depending on skepticism\n",title)

  plot(
    f1,
    xaxt="n",
    yaxt="n",
    main = main,
    cex.lab=1.17,
    xlab = "Skepticism",
    ylab = "Evaluation Metrics",
    col="white",
    ylim=c(0,1)
  )
  axis(1, at=seq(1, 10, 1), cex.axis=0.85)
  axis(2, at=seq(0, 1, .1), cex.axis=0.85)


  # add point at maximum f_latency

  print(run)
  print(f_latency)
  print(f_latency[5])
  # max_index = which(f_latency==max(f_latency))
  # points(max_index, f_latency[max_index], pch=1, cex=3.5, col=cols[5])
  # max_index = which(f1==max(f1))
  # points(max_index, f1[max_index], pch=1, cex=3.5, col=cols[1])

  # add lines

  ys =  list  ( f1          , precision , recall , speed    , f_latency)
  legends = c ("F1"   ,  "Precision" , "Recall",  "Speed"  , "F_latency")
  cols =   c  ( "black"     , "blue"    , "red"  , "orange" , "darkgreen")
  ltys =   c  ( 1           , 2         , 5      , 1        , 1)
  pchs =   c  (  1          ,  1        ,  1     ,  1       ,  1)
  types =  c  ( "b"         , "b"       , "b"    , "b"      , "b")

  for (i in 1:length(ys)) {
    lines(ys[[i]], col=cols[i], lty=ltys[i], pch=" ", cex=1, type=types[i])
    label = round(ys[[i]],2)
    text(1:10, ys[[i]], label, cex=.7, col=cols[i])
  }

  # add legend

  legend(7, 0.3, legend=legends, col=cols, lty=ltys, pch=pchs, text.width=2, cex=1)

  if (savePDF) {
    print(savePath)
    dev.copy(pdf, savePath, height=5.5, width=7.75)
    dev.off()
  }

}
