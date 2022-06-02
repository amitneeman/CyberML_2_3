library(stringr)
library(latex2exp)
#library(scales)
library(ggplot2)

BASE_DIR = "~/sshfs/code/eSPD-lab/"

configs = list(
  # BASE
  c(
    "segment_accuracy_by_num_of_messages/skepticism-5/",
    "2020-11-21_23-11-03__bert_classifier_on_PANC_with_seq-len-512",
    "non_quantized",
    "base (non quantized) on PANC"
  )
  , c(
    "segment_accuracy_by_num_of_messages/skepticism-5/",
    "2020-11-22_00-45-54__bert_classifier_on_PANC_with_seq-len-512",
    "non_quantized",
    "base (non quantized) on PANC"
  )
  , c(
    "segment_accuracy_by_num_of_messages/skepticism-5/",
    "2020-11-30_15-06-54__bert_classifier_on_PANC_with_seq-len-512",
    "non_quantized",
    "base (non quantized) on PANC"
  )

  # MOBILE
  , c(
    "segment_accuracy_by_num_of_messages/skepticism-5/",
    "2020-11-30_15-59-17__mobilebert_classifier_on_PANC_with_seq-len-512",
    "non_quantized",
    "MobileBERT (non quantized) on PANC"
  )
  , c(
    "segment_accuracy_by_num_of_messages/skepticism-5/",
    "2020-11-30_17-08-16__mobilebert_classifier_on_PANC_with_seq-len-512",
    "non_quantized",
    "MobileBERT (non quantized) on PANC"
  )
  , c(
    "segment_accuracy_by_num_of_messages/skepticism-5/",
    "2020-11-30_17-32-51__mobilebert_classifier_on_PANC_with_seq-len-512",
    "non_quantized",
    "MobileBERT (non quantized) on PANC"
  )

  # LARGE
  , c(
    "segment_accuracy_by_num_of_messages/skepticism-5/",
    "2020-12-27_04-09-08__bert-large-uncased_on_PANC_with_seq-len-512",
    "non_quantized",
    "large (non quantized) on PANC"
  )
  , c(
    "segment_accuracy_by_num_of_messages/skepticism-5/",
    "2020-12-27_11-56-28__bert-large-uncased_on_PANC_with_seq-len-512",
    "non_quantized",
    "large (non quantized) on PANC"
  )
  , c(
    "segment_accuracy_by_num_of_messages/skepticism-5/",
    "2020-12-27_15-59-58__bert-large-uncased_on_PANC_with_seq-len-512",
    "non_quantized",
    "large (non quantized) on PANC"
  )
)

savePDF = FALSE

for (config in configs) {

  eval_dir=config[1]
  run=config[2]
  skepticism_str=str_sub(eval_dir,end=-2)
  savePath=paste(sep="",BASE_DIR,"resources/",run,"/",model_version,"/message_based_eval/",eval_dir,"scores_by_message_num-",skepticism_str,".pdf")
  note=config[4]
  print(eval_dir)
  print(run)
  print(savePath)
  print(note)

  f1_dat = readLines(paste(sep="", BASE_DIR,"resources/",run,"/non_quantized/message_based_eval/",eval_dir,"f1.txt"))
  f1 = as.numeric(f1_dat)

  precision_dat = readLines(paste(sep="", BASE_DIR,"resources/",run,"/non_quantized/message_based_eval/",eval_dir,"precision.txt"))
  precision = as.numeric(precision_dat)

  recall_dat = readLines(paste(sep="", BASE_DIR,"resources/",run,"/non_quantized/message_based_eval/",eval_dir,"recall.txt"))
  recall = as.numeric(recall_dat)

  samples_dat = readLines(paste(sep="", BASE_DIR,"resources/",run,"/non_quantized/message_based_eval/",eval_dir,"samples.txt"))
  samples = as.numeric(samples_dat)


  par(mfrow=c(1,1))
  par(mar=c(5, 4, 4, 6) + 0.1)

  no_of_samples = samples[1]

  xlim=150 # messages
  ylim=70000 # samples

  # samples plot with y-axis on right

  plot(samples[1:xlim], ylim=c(1,ylim), type="l", col="white", xlab="", ylab="", axes=FALSE)
  # lines(pos[1:xlim], ylim=c(1,ylim), col="darkorange")
  # lines(neg[1:xlim], ylim=c(1,ylim), col="violet")
  #lines(samples[1:xlim], ylim=c(1,ylim), col="pink")
  #lines(samples[1:xlim], ylim=c(1,ylim), col="pink")
  lines(samples[1:xlim], ylim=c(1,ylim), lty=6, col="violet")
  mtext("Remaining segments in test set",cex=1.17,side=4,col="black",line=4.5)
  # mtext("Number of segments considered\npositives (orange), negatives (violet)",side=4,col="black",line=4.5)
  axis(4, ylim=c(0,7000), col="black",col.axis="black",las=1)


  # metrics plot with y-axis on left

  # draw new plot on top of old one
  par(new=TRUE)

  main = paste(sep="",
               "Metrics depending on segment length\n",note,"\n",skepticism_str, " n=",no_of_samples)
  plot(f1[1:xlim], xaxt="n",
       main = main,
       cex.lab=1.17,
       xlab = "Number of Messages",
       ylab = "Evaluation Metrics",
       col="white",ylim=c(0,1))

  lines(f1[1:xlim], col="black", ylim=c(0,1))
  lines(precision[1:xlim], col="blue", lty=2, ylim=c(0,1))
  lines(recall[1:xlim], col="red", lty=5, ylim=c(0,1))
  axis(1, at=c(1, seq(10, xlim, 10)), cex.axis=0.85)


  # Legend

  types = c("l", "l", "l", "l")
  cols = c("blue", "red", "black", "violet")
  ltys = c(2,5,1,6)
  pchs = c(-1,-1,-1,-1)


  legend(99, 0.4, legend=c(
    "Precision",
    "Recall",
    "F1",
    "#{test segments}"
  ), col=cols, lty=ltys, pch=pchs, text.width=37, cex=1)

  if (savePDF) {
    # dev.copy(pdf, savePath, height=5.5, width=7.75)
    dev.copy(pdf, savePath, height=6, width=8.5)
    dev.off()
  }

}
