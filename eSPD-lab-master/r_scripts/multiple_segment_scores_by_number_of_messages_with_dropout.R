library(stringr)
library(latex2exp)
#library(scales)
library(ggplot2)

configs = list(
  c(
    "segment_accuracy_by_num_of_messages/skepticism-5/",
    "2020-12-27_04-09-08__bert-large-uncased_on_PANC_with_seq-len-512",
    "non_quantized",
    "BERT-large (non quantized) on PANC"
  ),
  c(
    "segment_accuracy_by_num_of_messages/skepticism-5/",
    "2020-11-22_00-45-54__bert_classifier_on_PANC_with_seq-len-512",
    "non_quantized",
    "BERT-basic (non quantized) on PANC"
  ),
  c(
    "segment_accuracy_by_num_of_messages/skepticism-5/",
    "2020-11-30_15-59-17__mobilebert_classifier_on_PANC_with_seq-len-512",
    "non_quantized",
    "MobileBERT (non quantized) on PANC"
  )
)

savePDF = TRUE
BASE_DIR = "~/sshfs/code/eSPD-lab/"

alreadyPlotted = FALSE

# Legend
types = c("l", "l", "l")
cols = c("darkorange", "#40b70c", "blue", "violet")
ltys = c(1,1,1,6)
pchs = c(-1,-1,-1,-1)

for (i in 1:length(configs)) {
  config=configs[[i]]
  eval_dir=config[1]
  run=config[2]
  skepticism_str=str_sub(eval_dir,end=-2)
  savePath = paste(sep="", BASE_DIR, "r_scripts/plots/multiple_scores_by_message_length.pdf")
  print(savePath)
  note=config[4]
  print(eval_dir)
  print(run)
  print(savePath)
  print(note)

  f1_dat = readLines(paste(sep="", BASE_DIR, "/resources/",run,"/non_quantized/message_based_eval/",eval_dir,"f1.txt"))
  f1 = as.numeric(f1_dat)

  samples_dat = readLines(paste(sep="", BASE_DIR, "/resources/",run,"/non_quantized/message_based_eval/",eval_dir,"samples.txt"))
  samples = as.numeric(samples_dat)


  par(mfrow=c(1,1))
  par(mar=c(5, 4, 4, 6) + 0.1)

  no_of_samples = samples[1]

  xlim=150 # messages
  ylim=70000 # samples

  if (!alreadyPlotted) {
    #
    # samples plot with y-axis on right
    #

    plot(samples[1:xlim], ylim=c(1,ylim), type="l", col="white", xlab="", ylab="", axes=FALSE)
    mtext("Remaining segments in test set",cex=1.17,side=4,col="black",line=4.5)
    # mtext("Number of segments considered\npositives (orange), negatives (violet)",side=4,col="black",line=4.5)
    axis(4, ylim=c(0,7000), col="black",col.axis="black",las=1)
    lines(samples[1:xlim], ylim=c(1,ylim), lty=ltys[4], col=cols[4])

    #
    # metrics plot with y-axis on left
    #

    par(new=TRUE) # draw new plot on top of old one

    main = paste(sep="",
                 "F1 depending on number of messages for our detectors\non the chat segments in the dataset PANC")
    plot(f1[1:xlim], xaxt="n",
         main = main,
         cex.lab=1.17,
         xlab = "Number of Messages",
         ylab = "F1",
         col="white",ylim=c(0,1))

    axis(1, at=c(1, seq(10, xlim, 10)), cex.axis=0.85)

    alreadyPlotted = TRUE
  }

  lines(f1[1:xlim], lty=ltys[i], col=cols[i], ylim=c(0,1))
}


legend(99, 0.4, legend=c(
  "BERT-large",
  "BERT-base",
  # expression(BERT[large]),
  # expression(BERT[base]),
  "MobileBERT",
  "#{test segments}"
), col=cols, lty=ltys, pch=pchs, text.width=37, cex=1)

if (savePDF) {
  dev.copy(pdf, savePath, height=6, width=8.5)
  dev.off()
}
