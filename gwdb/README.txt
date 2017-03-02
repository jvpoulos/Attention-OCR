======= TERMS OF USAGE ======= 

The Washington database may be used for non-commercial research and teaching purposes only. If you are publishing scientific work based on the Washington database, we request you to include a reference to:

A. Fischer, A. Keller, V. Frinken, and H. Bunke: "Lexicon-Free Handwritten Word Spotting Using Character HMMs," in Pattern Recognition Letters, Volume 33(7), pages 934-942, 2012.

======= DATA SET =======

* IDs

The word ID "270-01-01" can be read as follows: page 270 (Library of Congress, George Washington Papers, Series 2, Letterbook 1), line 1, word 1. The line ID is "270-01" and the page ID is "270" accordingly. Word and line numbers start at 1.

* sets/

Keywords alongside with disjoint sets for training, validation, and testing  (four cross validations) given by line IDs used, for example, in:

- A. Fischer, A. Keller, V. Frinken, and H. Bunke: "Lexicon-Free Handwritten Word Spotting Using Character HMMs," in Pattern Recognition Letters, Volume 33(7), pages 934-942, 2012.

- V. Frinken, A. Fischer, R. Manmatha, and H. Bunke: "A Novel Word Spotting Method Based on Recurrent Neural Networks," in IEEE Trans. PAMI, Volume 34(2), pages 211-224, 2012.

* data/line_images_normalized

PNG images of binarized and normalized (skew and height) text lines.

* data/word_images_normalized

PNG images of binarized and normalized (skew and height) words.

* ground_truth/transcription.txt

[lineID] [word_1]|...|[word_n]

Text line transcription with word spellings given by:

[character_1]-...-[character_m]

Besides lower case and upper case letters, several punctuation marks (period "s_pt", comma "s_cm", etc) are used as well as special characters that are specific to the old-fashioned English longhand script (long s "s_s", first "s_1st", etc).

* ground_truth/word_labels.txt

Word image labels.

======= CONTACT INFORMATION ======= 

If you have any questions or suggestions, please contact Andreas Fischer (afischer@iam.unibe.ch).
