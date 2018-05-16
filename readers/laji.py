        # if result_dir is not None and result_prefix is not None:
        #     result_file = os.path.join(result_dir, result_prefix + '.json')
        #     with open(result_file, 'w') as fout:
        #         for pred_answer in pred_answers:
        #             fout.write(json.dumps(
        #                 pred_answer, ensure_ascii=False) + '\n')

            # self.logger.info('Saving {} results to {}'.format(
            #     result_prefix, result_file))

                # p_emb, q_emb, \
                #     sep_q_encodes, \
                #     sep_p_encodes, \
                #     match_p_encodes, \
                #     fuse_p_encodes, \
                #     concat_passage_encodes, \
                #     no_dup_question_encodes = self.sess.run([self.p_emb,
                #                                              self.q_emb,
                #                                              self.sep_q_encodes,
                #                                              self.sep_p_encodes,
                #                                              self.match_p_encodes,
                #                                              self.fuse_p_encodes,
                #                                              self.concat_passage_encodes,
                #                                              self.no_dup_question_encodes], feed_dict)
                # print('p_emb:', p_emb.shape)
                # print('q_emb:', q_emb.shape)
                # print('sep_q_encodes:', sep_q_encodes.shape)
                # print('sep_p_encodes:', sep_p_encodes.shape)
                # print('match_p_encodes:', match_p_encodes.shape)
                # print('fuse_p_encodes:', fuse_p_encodes.shape)
                # print('concat_passage_encodes:', concat_passage_encodes.shape)
                # print('no_dup_question_encodes:', no_dup_question_encodes.shape)
                # print('********************************')

                self.p_emb = model.p_emb
        self.q_emb = model.q_emb
        self.sep_q_encodes = model.sep_q_encodes
        self.sep_p_encodes = model.sep_p_encodes
        self.match_p_encodes = model.match_p_encodes
        self.fuse_p_encodes = model.fuse_p_encodes
        self.concat_passage_encodes = model.concat_passage_encodes
        self.no_dup_question_encodes = model.no_dup_question_encodes
