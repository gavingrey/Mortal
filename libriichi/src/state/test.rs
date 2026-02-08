use super::{ActionCandidate, PlayerState};
use crate::algo::shanten;
use crate::consts::MAX_VERSION;
use crate::hand::{hand, hand_with_aka, tile37_to_vec};
use crate::mjai::Event;
use crate::{matches_tu8, must_tile, t, tuz};
use std::mem;

impl PlayerState {
    fn test_update(&mut self, event: &Event) -> ActionCandidate {
        let cans = self.update(event).unwrap();
        self.validate();
        cans
    }

    fn test_update_json(&mut self, mjai_json: &str) -> ActionCandidate {
        let cans = self.update_json(mjai_json).unwrap();
        self.validate();
        cans
    }

    fn from_log(player_id: u8, log: &str) -> Self {
        let mut ps = Self::new(player_id);
        for line in log.trim().split('\n') {
            ps.test_update_json(line);
        }
        ps
    }

    fn num_doras_in_hand(&self) -> u8 {
        self.tehai
            .iter()
            .zip(self.dora_factor.iter())
            .map(|(&count, &f)| count * f)
            .chain(self.akas_in_hand.iter().map(|&b| b as u8))
            .chain(
                self.fuuro_overview[0]
                    .iter()
                    .flatten()
                    .map(|t| self.dora_factor[t.deaka().as_usize()] + t.is_aka() as u8),
            )
            .chain(self.ankan_overview[0].iter().map(|t| {
                self.dora_factor[t.deaka().as_usize()] * 4
                    + matches_tu8!(t.as_u8(), 5m | 5p | 5s) as u8
            }))
            .sum()
    }

    fn validate(&self) {
        assert_eq!(
            self.real_time_shanten(),
            shanten::calc_all(&self.tehai, self.tehai_len_div3),
        );
        assert_eq!(
            self.is_menzen,
            self.chis.is_empty() && self.pons.is_empty() && self.minkans.is_empty()
        );
        assert_eq!(self.doras_owned[0], self.num_doras_in_hand());
        if self.last_cans.can_act() {
            for version in 1..=MAX_VERSION {
                let _encoded = self.encode_obs(version, false);
                if self.last_cans.can_kakan || self.last_cans.can_ankan {
                    let _encoded = self.encode_obs(version, true);
                }
            }
        }
    }
}

#[test]
fn waits() {
    let mut ps = PlayerState {
        tehai: hand("456m 78999p 789s 77z").unwrap(),
        tehai_len_div3: 4,
        ..Default::default()
    };
    ps.update_waits_and_furiten();
    let expected = t![6p, 9p, C];
    for (idx, &b) in ps.waits.iter().enumerate() {
        if expected.contains(&must_tile!(idx)) {
            assert!(b);
        } else {
            assert!(!b);
        }
    }

    let mut ps = PlayerState {
        tehai: hand("2344445666678s").unwrap(),
        tehai_len_div3: 4,
        ..Default::default()
    };
    ps.update_waits_and_furiten();
    let expected = t![1s, 2s, 3s, 5s, 7s, 8s, 9s];
    for (idx, &b) in ps.waits.iter().enumerate() {
        if expected.contains(&must_tile!(idx)) {
            assert!(b);
        } else {
            assert!(!b);
        }
    }
}

#[test]
fn can_chi() {
    let mut ps = PlayerState::new(0);
    ps.tehai = hand("1111234m").unwrap();
    ps.set_can_chi_from_tile(t!(1m));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,
            can_chi_mid: false,
            can_chi_low: false,
            ..
        },
    ));
    ps.set_can_chi_from_tile(t!(4m));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,
            can_chi_mid: false,
            can_chi_low: false,
            ..
        },
    ));
    ps.set_can_chi_from_tile(t!(2m));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,
            can_chi_mid: true,
            can_chi_low: true,
            ..
        },
    ));

    ps.tehai = hand("6666789999p").unwrap();
    ps.set_can_chi_from_tile(t!(5p));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,
            can_chi_mid: false,
            can_chi_low: true,
            ..
        },
    ));
    ps.set_can_chi_from_tile(t!(7p));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,
            can_chi_mid: true,
            can_chi_low: true,
            ..
        },
    ));
    ps.set_can_chi_from_tile(t!(8p));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: true,
            can_chi_mid: true,
            can_chi_low: false,
            ..
        },
    ));

    ps.tehai = hand("4556s").unwrap();
    ps.set_can_chi_from_tile(t!(3s));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,
            can_chi_mid: false,
            can_chi_low: true,
            ..
        },
    ));
    ps.set_can_chi_from_tile(t!(4s));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,
            can_chi_mid: false,
            can_chi_low: true,
            ..
        },
    ));
    ps.set_can_chi_from_tile(t!(5s));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: false,
            can_chi_mid: false,
            can_chi_low: false,
            ..
        },
    ));
    ps.set_can_chi_from_tile(t!(6s));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: true,
            can_chi_mid: false,
            can_chi_low: false,
            ..
        },
    ));
    ps.set_can_chi_from_tile(t!(7s));
    assert!(matches!(
        ps.last_cans,
        ActionCandidate {
            can_chi_high: true,
            can_chi_mid: false,
            can_chi_low: false,
            ..
        },
    ));
}

#[test]
fn furiten() {
    let mut ps = PlayerState::new(0);
    ps.test_update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(3p),
        tehais: [
            tile37_to_vec(&hand_with_aka("23406m 456789p 58s").unwrap())
                .try_into()
                .unwrap(),
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    });
    ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(8s),
    });
    assert!(ps.shanten == 1);
    assert!(ps.waits.iter().all(|&b| !b));
    ps.test_update(&Event::Dahai {
        actor: 0,
        pai: t!(5s),
        tsumogiri: false,
    });
    assert!(ps.shanten == 0);
    assert!(ps.waits[tuz!(1m)] && ps.waits[tuz!(4m)] && ps.waits[tuz!(7m)]);
    assert!(!ps.at_furiten);

    ps.test_update(&Event::Tsumo {
        actor: 1,
        pai: t!(?),
    });
    let cans = ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(1m),
        tsumogiri: false,
    });
    assert!(!ps.at_furiten);
    assert!(cans.can_ron_agari);

    ps.test_update(&Event::Tsumo {
        actor: 2,
        pai: t!(?),
    });
    assert!(ps.at_furiten);
    ps.test_update(&Event::Dahai {
        actor: 2,
        pai: t!(1s),
        tsumogiri: true,
    });

    ps.test_update(&Event::Tsumo {
        actor: 3,
        pai: t!(?),
    });
    let cans = ps.test_update(&Event::Dahai {
        actor: 3,
        pai: t!(1m),
        tsumogiri: false,
    });
    assert!(ps.shanten == 0);
    assert!(ps.waits[tuz!(1m)] && ps.waits[tuz!(4m)] && ps.waits[tuz!(7m)]);
    assert!(ps.at_furiten);
    assert!(!cans.can_ron_agari);

    ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(3s),
    });
    assert!(ps.at_furiten);
    ps.test_update(&Event::Dahai {
        actor: 0,
        pai: t!(3s),
        tsumogiri: true,
    });
    assert!(!ps.at_furiten);

    ps.test_update(&Event::Tsumo {
        actor: 1,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(P),
        tsumogiri: true,
    });

    ps.test_update(&Event::Tsumo {
        actor: 2,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 2,
        pai: t!(C),
        tsumogiri: true,
    });
    ps.test_update(&Event::Tsumo {
        actor: 3,
        pai: t!(?),
    });
    let cans = ps.test_update(&Event::Dahai {
        actor: 3,
        pai: t!(1m),
        tsumogiri: false,
    });
    assert!(!ps.at_furiten);
    assert!(cans.can_ron_agari);
    assert_eq!(ps.agari_points(true, &[]).unwrap().ron, 5800);

    // riichi furiten test
    let cans = ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(N),
    });
    assert!(cans.can_riichi);
    ps.test_update(&Event::Reach { actor: 0 });
    ps.test_update(&Event::Dahai {
        actor: 0,
        pai: t!(N),
        tsumogiri: true,
    });
    ps.test_update(&Event::ReachAccepted { actor: 0 });

    ps.test_update(&Event::Tsumo {
        actor: 1,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(9m),
        tsumogiri: true,
    });
    ps.test_update(&Event::Tsumo {
        actor: 2,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 2,
        pai: t!(9m),
        tsumogiri: true,
    });
    ps.test_update(&Event::Tsumo {
        actor: 3,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 3,
        pai: t!(9m),
        tsumogiri: true,
    });

    // tsumo agari minogashi
    let cans = ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(1m),
    });
    assert!(ps.waits[tuz!(1m)] && ps.waits[tuz!(4m)] && ps.waits[tuz!(7m)]);
    assert!(!ps.at_furiten);
    assert!(cans.can_tsumo_agari);
    ps.test_update(&Event::Dahai {
        actor: 0,
        pai: t!(1m),
        tsumogiri: true,
    });
    assert!(ps.at_furiten); // furiten forever from now on

    ps.test_update(&Event::Tsumo {
        actor: 1,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(4s),
        tsumogiri: true,
    });
    ps.test_update(&Event::Tsumo {
        actor: 2,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 2,
        pai: t!(4s),
        tsumogiri: true,
    });
    ps.test_update(&Event::Tsumo {
        actor: 3,
        pai: t!(?),
    });
    let cans = ps.test_update(&Event::Dahai {
        actor: 3,
        pai: t!(7m),
        tsumogiri: true,
    });
    assert!(ps.waits[tuz!(1m)] && ps.waits[tuz!(4m)] && ps.waits[tuz!(7m)]);
    assert!(ps.at_furiten);
    assert!(!cans.can_ron_agari);

    ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(8m),
    });
    ps.test_update(&Event::Dahai {
        actor: 0,
        pai: t!(8m),
        tsumogiri: true,
    });
    assert!(ps.at_furiten); // still furiten

    ps.test_update(&Event::Tsumo {
        actor: 1,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(E),
        tsumogiri: true,
    });
    ps.test_update(&Event::Tsumo {
        actor: 2,
        pai: t!(?),
    });
    let cans = ps.test_update(&Event::Dahai {
        actor: 2,
        pai: t!(4m),
        tsumogiri: true,
    });
    assert!(ps.at_furiten);
    assert!(!cans.can_ron_agari);
    ps.test_update(&Event::Tsumo {
        actor: 3,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 3,
        pai: t!(E),
        tsumogiri: true,
    });

    // tsumo agari is always possible regardless of furiten
    let cans = ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(4m),
    });
    assert!(ps.waits[0] && ps.waits[3] && ps.waits[6]);
    assert!(ps.at_furiten);
    assert!(cans.can_tsumo_agari);
    assert_eq!(ps.agari_points(false, &[t!(3m)]).unwrap().tsumo_ko, 6000);
}

#[test]
fn dora_count_after_kan() {
    let mut ps = PlayerState::new(0);
    ps.test_update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(N),
        tehais: [
            tile37_to_vec(&hand_with_aka("1111s 123456p 112z").unwrap())
                .try_into()
                .unwrap(),
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    });
    ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(8s),
    });
    assert_eq!(ps.doras_owned[0], 2);

    ps.test_update(&Event::Ankan {
        actor: 0,
        consumed: [t!(1s); 4],
    });
    ps.test_update(&Event::Dora {
        dora_marker: t!(9s),
    });
    ps.test_update(&Event::Tsumo {
        actor: 0,
        pai: t!(5pr),
    });
    assert_eq!(ps.doras_owned[0], 7);
    ps.test_update(&Event::Dahai {
        actor: 0,
        pai: t!(E),
        tsumogiri: true,
    });
    assert_eq!(ps.doras_owned[0], 6);

    ps.test_update(&Event::Tsumo {
        actor: 1,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(5p),
        tsumogiri: true,
    });

    ps.test_update(&Event::Pon {
        actor: 0,
        target: 1,
        pai: t!(5p),
        consumed: t![5pr, 5p],
    });
    assert_eq!(ps.doras_owned[0], 6);
    ps.test_update(&Event::Dahai {
        actor: 0,
        pai: t!(E),
        tsumogiri: false,
    });
    assert_eq!(ps.doras_owned[0], 5);

    ps.test_update(&Event::Tsumo {
        actor: 1,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(P),
        tsumogiri: true,
    });
    ps.test_update(&Event::Tsumo {
        actor: 2,
        pai: t!(?),
    });
    ps.test_update(&Event::Dahai {
        actor: 2,
        pai: t!(P),
        tsumogiri: true,
    });

    ps.test_update(&Event::Tsumo {
        actor: 3,
        pai: t!(?),
    });
    ps.test_update(&Event::Ankan {
        actor: 3,
        consumed: [t!(1m); 4],
    });
    ps.test_update(&Event::Dora {
        dora_marker: t!(4p),
    });
    assert_eq!(ps.doras_owned[0], 8);
}

#[test]
fn rule_based_agari_all_last_minogashi() {
    let log = r#"
        {"type":"start_kyoku","bakaze":"S","dora_marker":"5m","kyoku":4,"honba":0,"kyotaku":0,"oya":3,"scores":[35300,3000,38400,23300],"tehais":[["4m","5mr","8m","1p","3p","3p","5p","2s","5sr","9s","W","P","P"],["2m","3m","5m","7m","7p","9p","4s","5s","5s","6s","7s","7s","E"],["3m","5m","6m","2p","6p","9p","1s","5s","8s","9s","S","S","C"],["1m","4m","3p","4p","5pr","7p","1s","2s","7s","8s","W","N","P"]]}
        {"type":"tsumo","actor":3,"pai":"F"}
        {"type":"dahai","actor":3,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"5p"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"9m"}
        {"type":"dahai","actor":1,"pai":"E","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"N"}
        {"type":"dahai","actor":2,"pai":"9p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"2p"}
        {"type":"dahai","actor":3,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"6m"}
        {"type":"dahai","actor":0,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"7m"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"3s"}
        {"type":"dahai","actor":2,"pai":"2p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"4s"}
        {"type":"dahai","actor":3,"pai":"W","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1m"}
        {"type":"dahai","actor":0,"pai":"1m","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"9m"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"3m"}
        {"type":"dahai","actor":2,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"2s"}
        {"type":"dahai","actor":3,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"2m"}
        {"type":"dahai","actor":0,"pai":"2s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"1m"}
        {"type":"dahai","actor":1,"pai":"5m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"3p"}
        {"type":"dahai","actor":2,"pai":"3p","tsumogiri":true}
        {"type":"pon","actor":0,"target":2,"pai":"3p","consumed":["3p","3p"]}
        {"type":"dahai","actor":0,"pai":"2m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"6p"}
        {"type":"dahai","actor":1,"pai":"9p","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"6s"}
        {"type":"dahai","actor":2,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"7p"}
        {"type":"dahai","actor":3,"pai":"P","tsumogiri":false}
        {"type":"pon","actor":0,"target":3,"pai":"P","consumed":["P","P"]}
        {"type":"dahai","actor":0,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"7s"}
        {"type":"dahai","actor":1,"pai":"5s","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"3s"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"2m"}
        {"type":"dahai","actor":3,"pai":"1s","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1p"}
        {"type":"dahai","actor":0,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"7m"}
        {"type":"dahai","actor":1,"pai":"4s","tsumogiri":false}
        {"type":"chi","actor":2,"target":1,"pai":"4s","consumed":["5s","6s"]}
        {"type":"dahai","actor":2,"pai":"6p","tsumogiri":false}
        {"type":"chi","actor":3,"target":2,"pai":"6p","consumed":["5pr","7p"]}
        {"type":"dahai","actor":3,"pai":"7p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1s"}
        {"type":"dahai","actor":0,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"1s"}
        {"type":"reach","actor":1}
        {"type":"dahai","actor":1,"pai":"1s","tsumogiri":true}
        {"type":"reach_accepted","actor":1}
        {"type":"tsumo","actor":2,"pai":"9s"}
        {"type":"dahai","actor":2,"pai":"8s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"4p"}
        {"type":"dahai","actor":3,"pai":"4p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4m"}
        {"type":"dahai","actor":0,"pai":"4m","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"1p"}
        {"type":"dahai","actor":1,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"8m"}
        {"type":"dahai","actor":2,"pai":"8m","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"C"}
        {"type":"dahai","actor":3,"pai":"C","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"2s"}
        {"type":"dahai","actor":0,"pai":"2s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"8p"}
    "#;
    let mut ps = PlayerState::from_log(1, log);

    assert!(ps.last_cans.can_tsumo_agari);
    let should_hora = ps.rule_based_agari();
    assert!(!should_hora);

    let orig_scores = mem::replace(&mut ps.scores, [9000, 30000, 30000, 30000]);
    let should_hora = ps.rule_based_agari();
    assert!(should_hora);
    ps.scores = orig_scores;

    ps.add_dora_indicator(t!(5m)).unwrap();
    let should_hora = ps.rule_based_agari();
    assert!(should_hora);

    let log = r#"
        {"type":"start_kyoku","bakaze":"S","dora_marker":"3s","kyoku":4,"honba":1,"kyotaku":0,"oya":3,"scores":[39000,25000,16900,19100],"tehais":[["1m","2m","3m","5mr","6m","8m","2p","2p","5pr","7s","8s","S","S"],["7m","9m","9m","6p","7p","1s","1s","3s","4s","6s","6s","S","P"],["3m","4m","5m","7m","4p","5p","5p","6p","8p","9p","5sr","5s","F"],["1m","2m","2m","6m","8m","1p","9p","3s","5s","6s","7s","E","W"]]}
        {"type":"tsumo","actor":3,"pai":"N"}
        {"type":"dahai","actor":3,"pai":"9p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1s"}
        {"type":"dahai","actor":0,"pai":"5pr","tsumogiri":false}
        {"type":"pon","actor":2,"target":0,"pai":"5pr","consumed":["5p","5p"]}
        {"type":"dahai","actor":2,"pai":"9p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"7m"}
        {"type":"dahai","actor":3,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"C"}
        {"type":"dahai","actor":0,"pai":"8m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"7m"}
        {"type":"dahai","actor":1,"pai":"6p","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"9s"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"F"}
        {"type":"dahai","actor":3,"pai":"W","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"6m"}
        {"type":"dahai","actor":0,"pai":"1s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"2m"}
        {"type":"dahai","actor":1,"pai":"7p","tsumogiri":false}
        {"type":"chi","actor":2,"target":1,"pai":"7p","consumed":["6p","8p"]}
        {"type":"dahai","actor":2,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"4p"}
        {"type":"dahai","actor":3,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"6m"}
        {"type":"dahai","actor":1,"pai":"2m","tsumogiri":false}
        {"type":"pon","actor":3,"target":1,"pai":"2m","consumed":["2m","2m"]}
        {"type":"dahai","actor":3,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"4s"}
        {"type":"dahai","actor":0,"pai":"4s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"1s"}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"8s"}
        {"type":"dahai","actor":2,"pai":"8s","tsumogiri":true}
        {"type":"chi","actor":3,"target":2,"pai":"8s","consumed":["6s","7s"]}
        {"type":"dahai","actor":3,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"3p"}
        {"type":"dahai","actor":0,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"6p"}
        {"type":"dahai","actor":1,"pai":"6p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"4s"}
        {"type":"dahai","actor":2,"pai":"4p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"N"}
        {"type":"dahai","actor":3,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"3s"}
        {"type":"dahai","actor":0,"pai":"3s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"5s"}
        {"type":"dahai","actor":1,"pai":"S","tsumogiri":false}
        {"type":"pon","actor":0,"target":1,"pai":"S","consumed":["S","S"]}
        {"type":"dahai","actor":0,"pai":"3p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"3m"}
        {"type":"dahai","actor":1,"pai":"3m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"4p"}
        {"type":"dahai","actor":2,"pai":"4p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"P"}
        {"type":"dahai","actor":3,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"8p"}
        {"type":"dahai","actor":0,"pai":"8p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"4p"}
        {"type":"dahai","actor":1,"pai":"4p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"E"}
        {"type":"dahai","actor":2,"pai":"E","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"C"}
        {"type":"dahai","actor":3,"pai":"4p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"7p"}
        {"type":"dahai","actor":0,"pai":"7p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"8p"}
        {"type":"dahai","actor":1,"pai":"8p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"S"}
        {"type":"dahai","actor":2,"pai":"S","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"N"}
        {"type":"dahai","actor":3,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"2s"}
        {"type":"dahai","actor":0,"pai":"2s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"8s"}
        {"type":"dahai","actor":1,"pai":"8s","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"E"}
        {"type":"dahai","actor":2,"pai":"E","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"6s"}
        {"type":"dahai","actor":3,"pai":"E","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"9m"}
        {"type":"dahai","actor":0,"pai":"9m","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"F"}
        {"type":"dahai","actor":1,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"C"}
        {"type":"dahai","actor":2,"pai":"C","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"E"}
        {"type":"dahai","actor":3,"pai":"E","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"P"}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"N"}
        {"type":"dahai","actor":2,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"8m"}
        {"type":"dahai","actor":3,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"P"}
        {"type":"dahai","actor":0,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"4m"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"5m"}
        {"type":"dahai","actor":2,"pai":"4s","tsumogiri":false}
        {"type":"chi","actor":3,"target":2,"pai":"4s","consumed":["5s","6s"]}
        {"type":"dahai","actor":3,"pai":"3s","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1m"}
        {"type":"dahai","actor":0,"pai":"1m","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"8s"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"9s"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"7s"}
        {"type":"dahai","actor":3,"pai":"7s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"7s"}
        {"type":"dahai","actor":0,"pai":"6m","tsumogiri":false}
    "#;
    let ps = PlayerState::from_log(2, log);
    assert!(ps.rule_based_agari());
}

#[test]
fn get_rank() {
    let ps = PlayerState::new(0);
    let rank = ps.get_rank([20000, 25000, 25000, 30000]);
    assert_eq!(rank, 3);

    let ps = PlayerState::new(3);
    let rank = ps.get_rank([25000, 25000, 25000, 25000]);
    assert_eq!(rank, 3);

    let ps = PlayerState::new(1);
    let rank = ps.get_rank([25000, 30000, 20000, 25000]);
    assert_eq!(rank, 2);

    let ps = PlayerState::new(1);
    let rank = ps.get_rank([32000, 32000, 18000, 18000]);
    assert_eq!(rank, 0);

    let ps = PlayerState::new(2);
    let rank = ps.get_rank([32000, 18000, 18000, 32000]);
    assert_eq!(rank, 1);

    let ps = PlayerState::new(2);
    let rank = ps.get_rank([5, 2, 5, 3]);
    assert_eq!(rank, 1);
}

#[test]
fn kakan_from_hand() {
    let log = r#"
        {"type":"start_kyoku","bakaze":"S","dora_marker":"6m","kyoku":2,"honba":0,"kyotaku":0,"oya":1,"scores":[16100,36600,16800,30500],"tehais":[["5p","5s","1s","9m","9m","W","E","N","1p","F","9m","3p","6p"],["4s","9s","S","4s","1m","P","N","7s","F","2m","3s","2s","2s"],["6m","8p","8p","2p","8m","N","7p","C","1s","2p","N","9s","9p"],["2m","6s","7p","9s","2m","9s","6m","7s","8m","3m","S","5mr","C"]]}
        {"type":"tsumo","actor":1,"pai":"S"}
        {"type":"dahai","actor":1,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"1s"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"P"}
        {"type":"dahai","actor":3,"pai":"S","tsumogiri":false}
        {"type":"pon","actor":1,"target":3,"pai":"S","consumed":["S","S"]}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"4p"}
        {"type":"dahai","actor":2,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"5s"}
        {"type":"dahai","actor":3,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"7m"}
        {"type":"dahai","actor":0,"pai":"E","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"P"}
        {"type":"dahai","actor":1,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"9p"}
        {"type":"dahai","actor":2,"pai":"6m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"C"}
        {"type":"dahai","actor":3,"pai":"C","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"7p"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"5s"}
        {"type":"dahai","actor":1,"pai":"2m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"5m"}
        {"type":"dahai","actor":2,"pai":"5m","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"1p"}
        {"type":"dahai","actor":3,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4m"}
        {"type":"dahai","actor":0,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"E"}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"1s"}
        {"type":"dahai","actor":2,"pai":"8m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"6p"}
        {"type":"dahai","actor":3,"pai":"8m","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"5p"}
        {"type":"dahai","actor":0,"pai":"1s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"2s"}
        {"type":"dahai","actor":1,"pai":"E","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"5m"}
        {"type":"dahai","actor":2,"pai":"5m","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"3s"}
        {"type":"dahai","actor":3,"pai":"3s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"7p"}
        {"type":"dahai","actor":0,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"E"}
        {"type":"dahai","actor":1,"pai":"E","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"W"}
        {"type":"dahai","actor":2,"pai":"W","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"7m"}
        {"type":"dahai","actor":3,"pai":"2m","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"5m"}
        {"type":"dahai","actor":0,"pai":"5s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"S"}
        {"type":"dahai","actor":1,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"6p"}
        {"type":"dahai","actor":2,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"2p"}
        {"type":"dahai","actor":3,"pai":"2p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"6p"}
        {"type":"dahai","actor":0,"pai":"3p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"4m"}
        {"type":"dahai","actor":1,"pai":"4m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"3s"}
        {"type":"dahai","actor":2,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"8p"}
        {"type":"reach","actor":3}
        {"type":"dahai","actor":3,"pai":"P","tsumogiri":false}
        {"type":"reach_accepted","actor":3}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"8s"}
        {"type":"kakan","actor":1,"pai":"S","consumed":["S","S","S"]}
        {"type":"tsumo","actor":1,"pai":"4s"}
    "#;
    let ps = PlayerState::from_log(1, log);

    assert!(ps.last_cans.can_tsumo_agari);
}

#[test]
fn discard_candidates_with_unconditional_tenpai() {
    let log = r#"
        {"type":"start_kyoku","bakaze":"S","dora_marker":"2s","kyoku":3,"honba":0,"kyotaku":0,"oya":2,"scores":[25600,15600,21200,37600],"tehais":[["3m","3m","1p","6p","7p","9p","5sr","7s","8s","8s","E","E","W"],["4m","5mr","6m","1p","4p","5p","8p","3s","3s","4s","5s","S","P"],["1m","5m","7m","2p","9p","3s","5s","9s","S","W","N","P","C"],["1m","4m","6m","2p","3p","4p","6p","9p","2s","4s","7s","S","N"]]}
        {"type":"tsumo","actor":2,"pai":"C"}
        {"type":"dahai","actor":2,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"2m"}
        {"type":"dahai","actor":3,"pai":"2m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"2p"}
        {"type":"dahai","actor":0,"pai":"9p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"7p"}
        {"type":"dahai","actor":1,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"4p"}
        {"type":"dahai","actor":2,"pai":"W","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"P"}
        {"type":"dahai","actor":3,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"6m"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"C"}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"8m"}
        {"type":"dahai","actor":2,"pai":"9p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"9m"}
        {"type":"dahai","actor":3,"pai":"9m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"1p"}
        {"type":"dahai","actor":0,"pai":"2p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"7m"}
        {"type":"dahai","actor":1,"pai":"S","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"P"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"N"}
        {"type":"dahai","actor":3,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"6p"}
        {"type":"dahai","actor":0,"pai":"7p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"9m"}
        {"type":"dahai","actor":1,"pai":"C","tsumogiri":false}
        {"type":"pon","actor":2,"target":1,"pai":"C","consumed":["C","C"]}
        {"type":"dahai","actor":2,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"7s"}
        {"type":"dahai","actor":3,"pai":"7s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"2p"}
        {"type":"dahai","actor":0,"pai":"2p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"5pr"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":false}
        {"type":"chi","actor":2,"target":1,"pai":"9m","consumed":["7m","8m"]}
        {"type":"dahai","actor":2,"pai":"S","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"E"}
        {"type":"dahai","actor":3,"pai":"E","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"5m"}
        {"type":"dahai","actor":0,"pai":"7s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"3p"}
        {"type":"dahai","actor":1,"pai":"5p","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"F"}
        {"type":"dahai","actor":2,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"2s"}
        {"type":"dahai","actor":3,"pai":"2s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4s"}
        {"type":"dahai","actor":0,"pai":"4s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"1p"}
        {"type":"dahai","actor":1,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"6s"}
        {"type":"dahai","actor":2,"pai":"5m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"6p"}
        {"type":"dahai","actor":3,"pai":"6p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"9p"}
        {"type":"dahai","actor":0,"pai":"9p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"5p"}
        {"type":"dahai","actor":1,"pai":"5p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"5s"}
        {"type":"dahai","actor":2,"pai":"5s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"9s"}
        {"type":"dahai","actor":3,"pai":"9s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"8m"}
        {"type":"dahai","actor":0,"pai":"8m","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"9m"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"9s"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"1s"}
        {"type":"dahai","actor":3,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"2m"}
        {"type":"dahai","actor":0,"pai":"5m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"8m"}
        {"type":"dahai","actor":1,"pai":"8m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"8p"}
        {"type":"dahai","actor":2,"pai":"8p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"7m"}
        {"type":"dahai","actor":3,"pai":"7m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"7p"}
        {"type":"dahai","actor":0,"pai":"7p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"8p"}
        {"type":"dahai","actor":1,"pai":"7m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"3m"}
        {"type":"dahai","actor":2,"pai":"3m","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"1s"}
        {"type":"dahai","actor":3,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4p"}
        {"type":"dahai","actor":0,"pai":"2m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"F"}
        {"type":"dahai","actor":1,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"9s"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"7m"}
        {"type":"dahai","actor":3,"pai":"7m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"F"}
        {"type":"dahai","actor":0,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"8s"}
        {"type":"dahai","actor":1,"pai":"8s","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"F"}
        {"type":"dahai","actor":2,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"1m"}
        {"type":"dahai","actor":3,"pai":"1m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"9m"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"2m"}
        {"type":"dahai","actor":2,"pai":"2m","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"7p"}
        {"type":"dahai","actor":3,"pai":"7p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"3p"}
        {"type":"dahai","actor":0,"pai":"6m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"6m"}
        {"type":"dahai","actor":1,"pai":"6m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"1s"}
        {"type":"dahai","actor":2,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"8m"}
        {"type":"dahai","actor":3,"pai":"8m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"S"}
        {"type":"dahai","actor":0,"pai":"S","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"2m"}
        {"type":"dahai","actor":1,"pai":"2m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"4s"}
        {"type":"dahai","actor":2,"pai":"6s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"8s"}
        {"type":"dahai","actor":3,"pai":"8s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"N"}
        {"type":"dahai","actor":0,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"3s"}
    "#;
    let ps = PlayerState::from_log(1, log);

    let expected = t![7p, 8p];
    ps.discard_candidates_with_unconditional_tenpai()
        .iter()
        .enumerate()
        .for_each(|(idx, &b)| {
            if expected.contains(&must_tile!(idx)) {
                assert!(b);
            } else {
                assert!(!b);
            }
        });

    let log = r#"
        {"type":"start_kyoku","bakaze":"E","dora_marker":"2p","kyoku":4,"honba":0,"kyotaku":0,"oya":3,"scores":[25000,20100,24000,30900],"tehais":[["1m","1m","4m","5m","5m","1p","4p","6p","7p","4s","5s","6s","S"],["5m","6p","7p","2s","3s","4s","4s","5s","7s","9s","S","C","C"],["2m","3m","6m","7m","9m","9m","1p","6p","1s","6s","9s","P","P"],["5mr","6m","8m","8m","2p","5p","7p","8p","9p","3s","9s","W","N"]]}
        {"type":"tsumo","actor":3,"pai":"C"}
        {"type":"dahai","actor":3,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"E"}
        {"type":"dahai","actor":0,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"2m"}
        {"type":"dahai","actor":1,"pai":"2m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"9s"}
        {"type":"dahai","actor":2,"pai":"1s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"8p"}
        {"type":"dahai","actor":3,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"P"}
        {"type":"dahai","actor":0,"pai":"E","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"3m"}
        {"type":"dahai","actor":1,"pai":"3m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"8s"}
        {"type":"dahai","actor":2,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"S"}
        {"type":"dahai","actor":3,"pai":"S","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"N"}
        {"type":"dahai","actor":0,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"5pr"}
        {"type":"dahai","actor":1,"pai":"5m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"1s"}
        {"type":"dahai","actor":2,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"9p"}
        {"type":"dahai","actor":3,"pai":"W","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"2p"}
        {"type":"dahai","actor":0,"pai":"P","tsumogiri":false}
        {"type":"pon","actor":2,"target":0,"pai":"P","consumed":["P","P"]}
        {"type":"dahai","actor":2,"pai":"6p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"3p"}
        {"type":"dahai","actor":3,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"7m"}
        {"type":"dahai","actor":0,"pai":"S","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"2m"}
        {"type":"dahai","actor":1,"pai":"2m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"3s"}
        {"type":"dahai","actor":2,"pai":"3s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"3p"}
        {"type":"dahai","actor":3,"pai":"3s","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"8s"}
        {"type":"dahai","actor":0,"pai":"7m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"F"}
        {"type":"dahai","actor":1,"pai":"S","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"E"}
        {"type":"dahai","actor":2,"pai":"6s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"4s"}
        {"type":"dahai","actor":3,"pai":"4s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"7s"}
        {"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"6s"}
        {"type":"dahai","actor":1,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"7m"}
        {"type":"dahai","actor":2,"pai":"8s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"6m"}
        {"type":"dahai","actor":3,"pai":"2p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"3p"}
        {"type":"dahai","actor":0,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"6p"}
        {"type":"dahai","actor":1,"pai":"6p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"N"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"2p"}
        {"type":"dahai","actor":3,"pai":"2p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4p"}
        {"type":"dahai","actor":0,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"F"}
        {"type":"dahai","actor":1,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"3m"}
        {"type":"dahai","actor":2,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"8p"}
        {"type":"dahai","actor":3,"pai":"5p","tsumogiri":false}
        {"type":"chi","actor":0,"target":3,"pai":"5p","consumed":["6p","7p"]}
        {"type":"dahai","actor":0,"pai":"4m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"1p"}
        {"type":"dahai","actor":1,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"5s"}
        {"type":"dahai","actor":2,"pai":"5s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"9m"}
        {"type":"dahai","actor":3,"pai":"9m","tsumogiri":true}
        {"type":"pon","actor":2,"target":3,"pai":"9m","consumed":["9m","9m"]}
        {"type":"dahai","actor":2,"pai":"E","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"7s"}
        {"type":"dahai","actor":3,"pai":"7s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"3m"}
        {"type":"dahai","actor":0,"pai":"3m","tsumogiri":true}
        {"type":"pon","actor":2,"target":0,"pai":"3m","consumed":["3m","3m"]}
        {"type":"dahai","actor":2,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"1s"}
        {"type":"dahai","actor":3,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"7p"}
        {"type":"dahai","actor":0,"pai":"7p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"9m"}
        {"type":"dahai","actor":1,"pai":"9m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"4m"}
        {"type":"dahai","actor":2,"pai":"2m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"P"}
        {"type":"dahai","actor":3,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"F"}
        {"type":"dahai","actor":1,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"8m"}
        {"type":"dahai","actor":2,"pai":"8m","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"7s"}
        {"type":"dahai","actor":3,"pai":"7s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4p"}
        {"type":"dahai","actor":0,"pai":"4p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"3p"}
        {"type":"dahai","actor":1,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"8s"}
        {"type":"dahai","actor":2,"pai":"8s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"2s"}
        {"type":"dahai","actor":3,"pai":"2s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4p"}
        {"type":"dahai","actor":0,"pai":"4p","tsumogiri":true}
        {"type":"chi","actor":1,"target":0,"pai":"4p","consumed":["3p","5pr"]}
        {"type":"dahai","actor":1,"pai":"7s","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"5p"}
        {"type":"dahai","actor":2,"pai":"5p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"1m"}
        {"type":"dahai","actor":3,"pai":"8p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"W","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"8s"}
        {"type":"dahai","actor":1,"pai":"8s","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"8p"}
        {"type":"dahai","actor":2,"pai":"8p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"F"}
        {"type":"dahai","actor":3,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1p"}
        {"type":"dahai","actor":0,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"1m"}
        {"type":"dahai","actor":1,"pai":"1m","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"5sr"}
        {"type":"dahai","actor":2,"pai":"7m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"9p"}
        {"type":"dahai","actor":3,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1s"}
        {"type":"dahai","actor":0,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"6s"}
    "#;
    let ps = PlayerState::from_log(1, log);

    let expected = t![5p, 8p];
    for (idx, &b) in ps.waits.iter().enumerate() {
        if expected.contains(&must_tile!(idx)) {
            assert!(b);
        } else {
            assert!(!b);
        }
    }

    let discard_candidates = ps.discard_candidates_with_unconditional_tenpai();
    assert_eq!(discard_candidates, [false; 34]);
}

#[test]
fn double_chankan_ron() {
    let log = r#"
        {"type":"start_kyoku","bakaze":"S","dora_marker":"2p","kyoku":2,"honba":0,"kyotaku":0,"oya":1,"scores":[44400,1600,25700,28300],"tehais":[["1m","5m","9m","9m","9m","3p","9p","8s","9s","W","W","N","C"],["7m","8m","3p","6p","8p","1s","1s","3s","6s","9s","E","F","C"],["3m","9m","2p","5p","8p","1s","2s","5s","6s","7s","S","F","C"],["2m","2m","5m","5mr","8m","1p","1p","7p","8p","3s","5s","8s","9s"]]}
        {"type":"tsumo","actor":1,"pai":"P"}
        {"type":"dahai","actor":1,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"3m"}
        {"type":"dahai","actor":2,"pai":"F","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"6m"}
        {"type":"dahai","actor":3,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"1s"}
        {"type":"dahai","actor":0,"pai":"1s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"9p"}
        {"type":"dahai","actor":1,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"9p"}
        {"type":"dahai","actor":2,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"7s"}
        {"type":"dahai","actor":3,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"7p"}
        {"type":"dahai","actor":0,"pai":"C","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"5m"}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"8s"}
        {"type":"dahai","actor":2,"pai":"9m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"7m"}
        {"type":"dahai","actor":3,"pai":"1p","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"1m","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"P"}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"4m"}
        {"type":"dahai","actor":2,"pai":"S","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"8m"}
        {"type":"dahai","actor":3,"pai":"8m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"8p"}
        {"type":"dahai","actor":0,"pai":"N","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"5sr"}
        {"type":"dahai","actor":1,"pai":"E","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"E"}
        {"type":"dahai","actor":2,"pai":"E","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"4p"}
        {"type":"dahai","actor":3,"pai":"4p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"1m"}
        {"type":"dahai","actor":0,"pai":"5m","tsumogiri":false}
        {"type":"pon","actor":3,"target":0,"pai":"5m","consumed":["5m","5mr"]}
        {"type":"dahai","actor":3,"pai":"8s","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"4s"}
        {"type":"dahai","actor":0,"pai":"4s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"N"}
        {"type":"dahai","actor":1,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"9p"}
        {"type":"dahai","actor":2,"pai":"8p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"C"}
        {"type":"dahai","actor":3,"pai":"C","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4s"}
        {"type":"dahai","actor":0,"pai":"4s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"1m"}
        {"type":"dahai","actor":1,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"4p"}
        {"type":"dahai","actor":2,"pai":"2p","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"P"}
        {"type":"dahai","actor":3,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"3m"}
        {"type":"dahai","actor":0,"pai":"3p","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"6s"}
        {"type":"dahai","actor":1,"pai":"9p","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"8s"}
        {"type":"dahai","actor":2,"pai":"3m","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"4m"}
        {"type":"dahai","actor":3,"pai":"4m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"P"}
        {"type":"dahai","actor":0,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"E"}
        {"type":"dahai","actor":1,"pai":"E","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"7s"}
        {"type":"dahai","actor":2,"pai":"2s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"F"}
        {"type":"dahai","actor":3,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4m"}
        {"type":"dahai","actor":0,"pai":"4m","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"2m"}
        {"type":"dahai","actor":1,"pai":"5m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"7p"}
        {"type":"dahai","actor":2,"pai":"7p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"2s"}
        {"type":"dahai","actor":3,"pai":"2s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"4p"}
        {"type":"dahai","actor":0,"pai":"4p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"5pr"}
        {"type":"dahai","actor":1,"pai":"8p","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"2s"}
        {"type":"dahai","actor":2,"pai":"2s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"F"}
        {"type":"dahai","actor":3,"pai":"F","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"6p"}
        {"type":"dahai","actor":0,"pai":"6p","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"7m"}
        {"type":"dahai","actor":1,"pai":"3p","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"1p"}
        {"type":"dahai","actor":2,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"9s"}
        {"type":"dahai","actor":3,"pai":"9s","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"S"}
        {"type":"dahai","actor":0,"pai":"S","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"7s"}
        {"type":"dahai","actor":1,"pai":"6s","tsumogiri":false}
        {"type":"chi","actor":2,"target":1,"pai":"6s","consumed":["5s","7s"]}
        {"type":"dahai","actor":2,"pai":"1s","tsumogiri":false}
        {"type":"pon","actor":1,"target":2,"pai":"1s","consumed":["1s","1s"]}
        {"type":"dahai","actor":1,"pai":"3s","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"2p"}
        {"type":"dahai","actor":2,"pai":"2p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"3p"}
        {"type":"dahai","actor":3,"pai":"3p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"6s"}
        {"type":"dahai","actor":0,"pai":"6s","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"6p"}
        {"type":"dahai","actor":1,"pai":"6p","tsumogiri":true}
        {"type":"chi","actor":2,"target":1,"pai":"6p","consumed":["4p","5p"]}
        {"type":"dahai","actor":2,"pai":"8s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"6m"}
        {"type":"dahai","actor":3,"pai":"3s","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"7m"}
        {"type":"dahai","actor":0,"pai":"8s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"6p"}
        {"type":"dahai","actor":1,"pai":"6p","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"5s"}
        {"type":"dahai","actor":2,"pai":"8s","tsumogiri":false}
        {"type":"tsumo","actor":3,"pai":"1p"}
        {"type":"dahai","actor":3,"pai":"1p","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"2s"}
        {"type":"dahai","actor":0,"pai":"9s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"1m"}
        {"type":"dahai","actor":1,"pai":"2m","tsumogiri":false}
        {"type":"pon","actor":3,"target":1,"pai":"2m","consumed":["2m","2m"]}
        {"type":"dahai","actor":3,"pai":"6m","tsumogiri":false}
        {"type":"tsumo","actor":0,"pai":"W"}
        {"type":"dahai","actor":0,"pai":"2s","tsumogiri":false}
        {"type":"tsumo","actor":1,"pai":"N"}
        {"type":"dahai","actor":1,"pai":"N","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"5p"}
        {"type":"dahai","actor":2,"pai":"5p","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"3m"}
        {"type":"dahai","actor":3,"pai":"3m","tsumogiri":true}
        {"type":"tsumo","actor":0,"pai":"6m"}
        {"type":"ankan","actor":0,"consumed":["W","W","W","W"]}
        {"type":"dora","dora_marker":"7p"}
        {"type":"tsumo","actor":0,"pai":"8m"}
        {"type":"dahai","actor":0,"pai":"6m","tsumogiri":false}
        {"type":"chi","actor":1,"target":0,"pai":"6m","consumed":["7m","8m"]}
        {"type":"dahai","actor":1,"pai":"7m","tsumogiri":false}
        {"type":"tsumo","actor":2,"pai":"3s"}
        {"type":"dahai","actor":2,"pai":"3s","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"2m"}
    "#;
    let mut ps = PlayerState::from_log(2, log);

    let mut ps_kakan = ps.clone();
    let cans = ps_kakan
        .test_update_json(r#"{"type":"kakan","actor":3,"pai":"2m","consumed":["2m","2m","2m"]}"#);
    assert!(cans.can_ron_agari);
    assert_eq!(ps_kakan.agari_points(true, &[]).unwrap().ron, 1000);

    let cans = ps.test_update_json(r#"{"type":"dahai","actor":3,"pai":"2m","tsumogiri":true}"#);
    assert!(!cans.can_ron_agari);
}

#[test]
fn chi_at_0_shanten() {
    let log = r#"
        {"type":"start_kyoku","bakaze":"E","dora_marker":"W","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","2m","3m","5p","5p","4s","5s","E","E","E","S","S","S"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"]]}
        {"type":"tsumo","actor":0,"pai":"P"}
        {"type":"dahai","actor":0,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":1,"pai":"?"}
        {"type":"dahai","actor":1,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":2,"pai":"?"}
        {"type":"dahai","actor":2,"pai":"P","tsumogiri":true}
        {"type":"tsumo","actor":3,"pai":"?"}
        {"type":"dahai","actor":3,"pai":"6s","tsumogiri":false}
    "#;
    let mut ps = PlayerState::from_log(0, log);

    assert_eq!(ps.shanten, 0);
    assert_eq!(ps.real_time_shanten(), 0);
    assert!(ps.last_cans.can_ron_agari);
    assert!(ps.last_cans.can_chi_high);

    ps.test_update_json(r#"{"type":"chi","actor":0,"target":3,"consumed":["4s","5s"],"pai":"6s"}"#);
    assert_eq!(ps.shanten, 0);
    assert_eq!(ps.real_time_shanten(), -1);
    assert!(ps.at_furiten);
    assert!(!ps.has_next_shanten_discard);
}

#[test]
fn patch_hand_updates_tiles_seen() {
    // Create a PlayerState for opponent 1 in replay mode.
    // In replay_mode, unknown hand tiles are silently ignored by witness_tile,
    // so the opponent's hand is NOT in tiles_seen. patch_hand should fix this.
    let mut ps1 = PlayerState::new(1);
    ps1.replay_mode = true;
    ps1.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(1m), // dora is 2m
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Opponent 1's tiles_seen should only have the dora indicator (1m)
    assert_eq!(ps1.tiles_seen[tuz!(1m)], 1); // dora indicator
    assert_eq!(ps1.tiles_seen[tuz!(2m)], 0); // no 2m seen yet

    // Patch with a hand containing 2m tiles (which are dora)
    let patch_tiles = t![2m, 2m, 3m, 4p, 5p, 6p, 7s, 8s, 9s, E, E, S, S];
    ps1.patch_hand(&patch_tiles);

    // After patching: tiles_seen should now include the patched tiles
    assert_eq!(ps1.tiles_seen[tuz!(1m)], 1); // still just dora indicator
    assert_eq!(ps1.tiles_seen[tuz!(2m)], 2); // 2 copies of 2m from hand
    assert_eq!(ps1.tiles_seen[tuz!(3m)], 1); // 1 copy of 3m from hand
    assert_eq!(ps1.tiles_seen[tuz!(4p)], 1);
    assert_eq!(ps1.tiles_seen[tuz!(E)], 2); // 2 copies of East
    assert_eq!(ps1.tiles_seen[tuz!(S)], 2); // 2 copies of South

    // doras_owned[0] should reflect 2m dora (dora_factor[2m] = 1, count = 2)
    assert_eq!(ps1.doras_owned[0], 2);

    // doras_seen should include all seen dora tiles
    // tiles_seen has 2 copies of 2m, dora_factor[2m] = 1, so doras_seen includes 2
    assert_eq!(ps1.doras_seen, 2);
}

#[test]
fn patch_hand_updates_doras_with_aka() {
    // Test that aka dora in patched hand is properly tracked
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;
    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(4p), // dora is 5p
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Patch with hand containing aka 5mr and regular 5p (both are dora via indicator)
    let patch_tiles = t![5mr, 5p, 6m, 7m, 8m, 1s, 2s, 3s, 9p, 9p, 9p, E, E];
    ps.patch_hand(&patch_tiles);

    // Check akas_in_hand
    assert!(ps.akas_in_hand[0]); // 5mr
    assert!(!ps.akas_in_hand[1]); // no 5pr
    assert!(!ps.akas_in_hand[2]); // no 5sr

    // Check akas_seen
    assert!(ps.akas_seen[0]); // 5mr seen

    // Check tiles_seen includes the hand tiles
    // 5mr.deaka() = 5m (index 4), 5p.deaka() = 5p (index 13)
    assert_eq!(ps.tiles_seen[tuz!(5m)], 1); // 5mr -> 5m
    assert_eq!(ps.tiles_seen[tuz!(5p)], 1); // 5p -> 5p
    assert_eq!(ps.tiles_seen[tuz!(4p)], 1); // dora indicator

    // doras_owned[0]:
    // - 5mr: dora_factor[5m] = 0 (dora indicator is 4p, so dora is 5p, not 5m)
    //   but 5mr is aka, so +1
    // - 5p: dora_factor[5p] = 1 (from 4p indicator -> 5p is dora)
    // Total: 0 + 1 + 1 = 2
    assert_eq!(ps.doras_owned[0], 2);

    // doras_seen:
    // tiles_seen[5p] * dora_factor[5p] = 1 * 1 = 1
    // plus 4p (dora indicator) tiles_seen[4p] * dora_factor[4p] = 1 * 0 = 0
    // plus aka 5mr seen: +1
    // Total: 1 + 1 = 2
    assert_eq!(ps.doras_seen, 2);
}

#[test]
fn patch_hand_after_replay_with_discards() {
    // More realistic test: replay some events, then patch.
    // Player 0 perspective, opponent 1 draws and discards, then we patch.
    let mut ps1 = PlayerState::new(1);
    ps1.replay_mode = true;
    ps1.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(3s), // dora is 4s
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 draws and discards (opponent 1 witnesses the discard)
    ps1.update(&Event::Tsumo {
        actor: 0,
        pai: t!(?),
    })
    .unwrap();
    ps1.update(&Event::Dahai {
        actor: 0,
        pai: t!(4s), // discards a dora tile
        tsumogiri: true,
    })
    .unwrap();

    // Opponent 1 draws (unknown) and discards
    ps1.update(&Event::Tsumo {
        actor: 1,
        pai: t!(?),
    })
    .unwrap();
    ps1.update(&Event::Dahai {
        actor: 1,
        pai: t!(N),
        tsumogiri: true,
    })
    .unwrap();

    // At this point, tiles_seen should have:
    // - dora indicator (3s): 1
    // - player 0's discard (4s): 1 (witnessed because actor 0 != self)
    // - opponent 1's own discard (N): NOT in tiles_seen.
    //   In replay_mode, self-discards go through move_tile (not witness_tile),
    //   and since the hand was all unknowns, move_tile is a no-op. The tile
    //   was also never witnessed at tsumo time (unknown tile skipped).
    assert_eq!(ps1.tiles_seen[tuz!(3s)], 1); // dora indicator
    assert_eq!(ps1.tiles_seen[tuz!(4s)], 1); // player 0's discard
    assert_eq!(ps1.tiles_seen[tuz!(N)], 0); // own discard not witnessed in replay_mode

    // Patch with 13 tiles (start 13, tsumo +1, dahai -1 = 13 still)
    let patch_tiles = t![1m, 2m, 3m, 4s, 4s, 6p, 7p, 8p, 1s, 2s, 3s, E, E];
    ps1.patch_hand(&patch_tiles);

    // tiles_seen is now recomputed from scratch in patch_hand:
    //   closed_hand + all_kawa + all_fuuro_consumed + all_ankan + dora_indicators
    //
    // Sources:
    //   hand: 1m(1), 2m(1), 3m(1), 4s(2), 6p(1), 7p(1), 8p(1), 1s(1), 2s(1), 3s(1), E(2)
    //   kawa[0] (self=player1): N(1)
    //   kawa[3] (player0 relative to player1): 4s(1)
    //   dora_indicators: 3s(1)
    assert_eq!(ps1.tiles_seen[tuz!(1m)], 1); // from hand
    assert_eq!(ps1.tiles_seen[tuz!(3s)], 1 + 1); // dora indicator + hand
    assert_eq!(ps1.tiles_seen[tuz!(4s)], 1 + 2); // player 0's discard + hand
    assert_eq!(ps1.tiles_seen[tuz!(N)], 1); // own discard now correctly in tiles_seen
    assert_eq!(ps1.tiles_seen[tuz!(E)], 2); // from hand

    // doras_owned[0]: 4s count in hand is 2, dora_factor[4s] = 1
    // No aka, so doras_owned[0] = 2
    assert_eq!(ps1.doras_owned[0], 2);

    // doras_seen: all tiles_seen with dora_factor
    // tiles_seen[4s] = 3, dora_factor[4s] = 1 -> 3
    // No akas seen -> total = 3
    assert_eq!(ps1.doras_seen, 3);
}

// ---- Comprehensive patch_hand tests ----

#[test]
fn patch_hand_tiles_seen_includes_own_discards() {
    // Set up a replay-mode PlayerState that has made discards.
    // After patch_hand, tiles_seen should include the discarded tiles
    // (they are in kawa_overview, which patch_hand scans).
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(9s),
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 draws and discards (to get to player 1's turn)
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(E), tsumogiri: true }).unwrap();

    // Player 1 (our replay player) draws and discards 3m
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(3m), tsumogiri: true }).unwrap();

    // Player 2, 3, 0 play a round
    ps.update(&Event::Tsumo { actor: 2, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 2, pai: t!(S), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 3, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 3, pai: t!(W), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(N), tsumogiri: true }).unwrap();

    // Player 1 draws again and discards 7p
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(7p), tsumogiri: true }).unwrap();

    // In replay mode, own discards might not be in tiles_seen
    // because the tiles came from unknown hand. Verify that.
    // kawa_overview[0] (self=player1) should have 3m and 7p.

    // Patch with some tiles
    let patch_tiles = t![1m, 2m, 4p, 5p, 6p, 7s, 8s, 9s, E, E, S, S, N];
    ps.patch_hand(&patch_tiles);

    // tiles_seen should now include own discards via kawa_overview
    assert!(ps.tiles_seen[tuz!(3m)] >= 1, "own discard 3m should be in tiles_seen");
    assert!(ps.tiles_seen[tuz!(7p)] >= 1, "own discard 7p should be in tiles_seen");

    // Also check other players' discards
    assert!(ps.tiles_seen[tuz!(E)] >= 1, "player 0's discard E should be in tiles_seen");
    assert!(ps.tiles_seen[tuz!(S)] >= 1, "player 2's discard S should be in tiles_seen");
    assert!(ps.tiles_seen[tuz!(W)] >= 1, "player 3's discard W should be in tiles_seen");
    assert!(ps.tiles_seen[tuz!(N)] >= 1, "player 0's 2nd discard N should be in tiles_seen");
}

#[test]
fn patch_hand_tiles_seen_includes_meld_consumed() {
    // Set up a PlayerState where player 1 has called chi, consuming
    // tiles from their hand. After patch, tiles_seen should include
    // the consumed tiles from the meld.
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(9s),
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 draws and discards 2m
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(2m), tsumogiri: true }).unwrap();

    // Player 1 calls chi on 2m with 3m + 4m
    ps.update(&Event::Chi {
        actor: 1,
        target: 0,
        pai: t!(2m),
        consumed: [t!(3m), t!(4m)],
    })
    .unwrap();

    // Player 1 discards after chi
    ps.update(&Event::Dahai { actor: 1, pai: t!(N), tsumogiri: false }).unwrap();

    // Continue play to get player 1 a tsumo
    ps.update(&Event::Tsumo { actor: 2, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 2, pai: t!(S), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 3, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 3, pai: t!(W), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(P), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();

    // Patch with 11 tiles (13 - 2 consumed by chi)
    let patch_tiles = t![1p, 2p, 3p, 7s, 8s, 9s, E, E, S, C, F];
    ps.patch_hand(&patch_tiles);

    // The chi meld consumed tiles (3m, 4m) should be in tiles_seen
    // 3m: from fuuro_overview consumed[0]
    // 4m: from fuuro_overview consumed[1]
    // 2m: the called tile is in kawa_overview of player 0 (the discarder)
    assert!(
        ps.tiles_seen[tuz!(3m)] >= 1,
        "chi consumed tile 3m should be in tiles_seen, got {}",
        ps.tiles_seen[tuz!(3m)]
    );
    assert!(
        ps.tiles_seen[tuz!(4m)] >= 1,
        "chi consumed tile 4m should be in tiles_seen, got {}",
        ps.tiles_seen[tuz!(4m)]
    );
    assert!(
        ps.tiles_seen[tuz!(2m)] >= 1,
        "chi called tile 2m should be in tiles_seen (via kawa), got {}",
        ps.tiles_seen[tuz!(2m)]
    );
}

#[test]
fn patch_hand_tiles_seen_includes_ankan() {
    // Player 0 has an ankan of 1m. After patch on player 1,
    // tiles_seen should include all 4 copies of 1m.
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(9s),
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 draws 1m, then declares ankan
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Ankan {
        actor: 0,
        consumed: [t!(1m); 4],
    })
    .unwrap();
    ps.update(&Event::Dora { dora_marker: t!(2p) }).unwrap();

    // Rinshan tsumo
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(E), tsumogiri: true }).unwrap();

    // Other players play
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(W), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 2, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 2, pai: t!(N), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 3, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 3, pai: t!(F), tsumogiri: true }).unwrap();

    // Player 0 draws again
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(S), tsumogiri: true }).unwrap();

    // Player 1 draws
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();

    // Patch player 1's hand
    let patch_tiles = t![2m, 3m, 4p, 5p, 6p, 7s, 8s, 9s, P, P, C, C];
    ps.patch_hand(&patch_tiles);

    // ankan_overview for player 0 (relative seat 3 from player 1's perspective)
    // should have 1m with 4 copies
    assert_eq!(
        ps.tiles_seen[tuz!(1m)], 4,
        "ankan of 1m should contribute 4 to tiles_seen, got {}",
        ps.tiles_seen[tuz!(1m)]
    );
}

#[test]
fn patch_hand_tiles_seen_includes_other_players_discards() {
    // After replay and patch, tiles_seen should include tiles
    // discarded by OTHER players (which go through witness_tile normally
    // during replay, but patch_hand also recomputes from kawa_overview).
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(9s),
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Players 0, 1, 2, 3 each draw and discard
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(1m), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(2m), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 2, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 2, pai: t!(3m), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 3, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 3, pai: t!(4m), tsumogiri: true }).unwrap();

    // Player 0 draws, discards
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(5m), tsumogiri: true }).unwrap();
    // Player 1 draws
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();

    let patch_tiles = t![6m, 7m, 8m, 1p, 2p, 3p, 1s, 2s, 3s, E, E, S, S];
    ps.patch_hand(&patch_tiles);

    // Other players' discards: 1m (p0), 3m (p2), 4m (p3), 5m (p0)
    // Own discard: 2m (p1)
    assert_eq!(ps.tiles_seen[tuz!(1m)], 1, "player 0's discard 1m");
    assert_eq!(ps.tiles_seen[tuz!(2m)], 1, "own discard 2m");
    assert_eq!(ps.tiles_seen[tuz!(3m)], 1, "player 2's discard 3m");
    assert_eq!(ps.tiles_seen[tuz!(4m)], 1, "player 3's discard 4m");
    assert_eq!(ps.tiles_seen[tuz!(5m)], 1, "player 0's 2nd discard 5m");

    // Hand tiles
    assert_eq!(ps.tiles_seen[tuz!(6m)], 1, "hand tile 6m");
    assert_eq!(ps.tiles_seen[tuz!(E)], 2, "hand tile E x2");
}

#[test]
fn patch_hand_tiles_seen_no_double_count() {
    // A tile appears in both own discards and other player's discards.
    // Verify tiles_seen counts each instance exactly once.
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(9s),
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 discards 5m
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(5m), tsumogiri: true }).unwrap();

    // Player 1 (us) also discards 5m
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(5m), tsumogiri: true }).unwrap();

    // Player 2 discards 5m too
    ps.update(&Event::Tsumo { actor: 2, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 2, pai: t!(5m), tsumogiri: true }).unwrap();

    // Player 3 plays
    ps.update(&Event::Tsumo { actor: 3, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 3, pai: t!(W), tsumogiri: true }).unwrap();

    // Player 0 plays
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(N), tsumogiri: true }).unwrap();

    // Player 1 draws
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();

    // Patch with hand containing one more 5m
    let patch_tiles = t![5m, 1p, 2p, 3p, 7s, 8s, 9s, E, E, S, S, C, C];
    ps.patch_hand(&patch_tiles);

    // 5m should appear exactly 4 times:
    //   1 from player 0's kawa
    //   1 from player 1's kawa (own discard)
    //   1 from player 2's kawa
    //   1 from hand
    assert_eq!(
        ps.tiles_seen[tuz!(5m)], 4,
        "5m should appear exactly 4 times in tiles_seen (3 discards + 1 in hand), got {}",
        ps.tiles_seen[tuz!(5m)]
    );
}

#[test]
fn patch_hand_tiles_seen_with_dora_indicators() {
    // Verify dora indicators are included in tiles_seen after patch.
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(3p), // dora is 4p
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 draws and plays
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(E), tsumogiri: true }).unwrap();
    // Player 1 draws
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();

    let patch_tiles = t![1m, 2m, 3m, 4p, 4p, 6s, 7s, 8s, E, E, S, S, N];
    ps.patch_hand(&patch_tiles);

    // Dora indicator 3p should be in tiles_seen
    assert_eq!(
        ps.tiles_seen[tuz!(3p)], 1,
        "dora indicator 3p should be in tiles_seen, got {}",
        ps.tiles_seen[tuz!(3p)]
    );

    // 4p is dora (dora_factor[4p] = 1), hand has 2 copies
    assert_eq!(ps.tiles_seen[tuz!(4p)], 2, "hand tile 4p x2");
    assert_eq!(ps.doras_owned[0], 2, "2 dora 4p in hand");
    // doras_seen = tiles_seen[4p] * dora_factor[4p] = 2
    assert_eq!(ps.doras_seen, 2);
}

#[test]
fn patch_hand_doras_owned_with_melds() {
    // Set up a hand where dora tiles are in open melds (not just closed hand).
    // Verify doras_owned counts both closed hand dora and meld dora.
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(4m), // dora is 5m
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 discards 5m (which is dora)
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(5m), tsumogiri: true }).unwrap();

    // Player 1 pons 5m with two 5m from hand
    ps.update(&Event::Pon {
        actor: 1,
        target: 0,
        pai: t!(5m),
        consumed: [t!(5m), t!(5m)],
    })
    .unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(W), tsumogiri: false }).unwrap();

    // Continue play
    ps.update(&Event::Tsumo { actor: 2, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 2, pai: t!(S), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 3, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 3, pai: t!(N), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(P), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();

    // Patch with hand containing another 5m (dora) in closed hand
    // Player 1 had 13 tiles, pon consumed 2, discard 1 = 10 tiles, tsumo +1 = 11
    let patch_tiles = t![5m, 6m, 7m, 1p, 2p, 3p, 7s, 8s, 9s, E, E];
    ps.patch_hand(&patch_tiles);

    // doras_owned[0]:
    // - Closed hand: 5m x1, dora_factor[5m] = 1 -> 1
    // - Pon meld: 5m x2 from consumed + 5m called = 3 tiles in fuuro
    //   All are 5m with dora_factor = 1 -> 3
    // Total = 1 + 3 = 4
    assert_eq!(
        ps.doras_owned[0], 4,
        "doras_owned should count both closed hand and meld dora, got {}",
        ps.doras_owned[0]
    );
}

#[test]
fn patch_hand_doras_seen_complete() {
    // After a full patch with discards, melds, and dora indicators,
    // verify doras_seen is the correct total across all tiles_seen.
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(4s), // dora is 5s
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 discards 5s (dora)
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(5s), tsumogiri: true }).unwrap();

    // Player 1 draws and discards 5s (another dora)
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(5s), tsumogiri: true }).unwrap();

    // Player 2, 3 play
    ps.update(&Event::Tsumo { actor: 2, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 2, pai: t!(E), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 3, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 3, pai: t!(S), tsumogiri: true }).unwrap();

    // Player 0 plays
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(W), tsumogiri: true }).unwrap();
    // Player 1 draws
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();

    // Patch with hand containing 5s (dora) in closed hand
    let patch_tiles = t![5s, 1m, 2m, 3m, 1p, 2p, 3p, 7s, 8s, 9s, N, N, C];
    ps.patch_hand(&patch_tiles);

    // doras_seen should count all 5s in tiles_seen:
    //   kawa: player 0 discarded 5s (1), player 1 discarded 5s (1)
    //   hand: 5s (1)
    //   total tiles_seen[5s] = 3, dora_factor[5s] = 1 -> 3
    //   No aka -> doras_seen = 3
    assert_eq!(ps.tiles_seen[tuz!(5s)], 3, "tiles_seen[5s] should be 3");
    assert_eq!(ps.doras_seen, 3, "doras_seen should be 3");
}

#[test]
fn patch_hand_akas_seen_from_discards() {
    // If an aka tile was discarded (appears in kawa_overview),
    // verify akas_seen includes it after patch.
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(9s),
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 discards aka 5mr
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(5mr), tsumogiri: true }).unwrap();

    // Player 1 draws
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();

    let patch_tiles = t![1m, 2m, 3m, 4p, 5p, 6p, 7s, 8s, 9s, E, E, S, S];
    ps.patch_hand(&patch_tiles);

    // akas_seen[0] (5mr) should be true because it's in kawa_overview
    assert!(ps.akas_seen[0], "5mr discarded by player 0 should set akas_seen[0]");
    // tiles_seen should count the deaka'd tile
    assert_eq!(
        ps.tiles_seen[tuz!(5m)], 1,
        "5mr discard should count as 5m in tiles_seen"
    );
    // doras_seen should include the aka bonus
    // No dora indicator for 5m, but aka is always +1
    assert_eq!(ps.doras_seen, 1, "aka 5mr should contribute 1 to doras_seen");
}

#[test]
fn patch_hand_empty_hand() {
    // Player has 4 open melds + kan = 0 closed tiles.
    // Patch with empty slice. Verify safe defaults.
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(9s),
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 discards, player 1 calls chi multiple times to empty hand.
    // Simulating 4 melds exactly is complex, so just set up minimal state
    // and patch with empty hand.

    // Player 0 discards 2m
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(2m), tsumogiri: true }).unwrap();

    // Player 1 chi
    ps.update(&Event::Chi {
        actor: 1, target: 0, pai: t!(2m), consumed: [t!(3m), t!(4m)],
    }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(E), tsumogiri: false }).unwrap();

    // Seat 2, 3, 0 play
    for actor in [2_u8, 3, 0] {
        ps.update(&Event::Tsumo { actor, pai: t!(?) }).unwrap();
        ps.update(&Event::Dahai { actor, pai: t!(N), tsumogiri: true }).unwrap();
    }

    // Player 1 draws, discards
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(S), tsumogiri: true }).unwrap();

    // Player 2 discards 6p, player 1 pons
    ps.update(&Event::Tsumo { actor: 2, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 2, pai: t!(6p), tsumogiri: true }).unwrap();
    ps.update(&Event::Pon {
        actor: 1, target: 2, pai: t!(6p), consumed: [t!(6p), t!(6p)],
    }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(W), tsumogiri: false }).unwrap();

    // More play
    for actor in [3_u8, 0] {
        ps.update(&Event::Tsumo { actor, pai: t!(?) }).unwrap();
        ps.update(&Event::Dahai { actor, pai: t!(P), tsumogiri: true }).unwrap();
    }

    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(C), tsumogiri: true }).unwrap();

    // Player 3 discards 9s, player 1 chi
    ps.update(&Event::Tsumo { actor: 2, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 2, pai: t!(7s), tsumogiri: true }).unwrap();
    ps.update(&Event::Chi {
        actor: 1, target: 2, pai: t!(7s), consumed: [t!(8s), t!(9s)],
    }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(F), tsumogiri: false }).unwrap();

    // More play
    for actor in [3_u8, 0] {
        ps.update(&Event::Tsumo { actor, pai: t!(?) }).unwrap();
        ps.update(&Event::Dahai { actor, pai: t!(C), tsumogiri: true }).unwrap();
    }

    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(1s), tsumogiri: true }).unwrap();

    // Player 0 discards 1p, player 1 pon
    ps.update(&Event::Tsumo { actor: 2, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 2, pai: t!(1p), tsumogiri: true }).unwrap();
    ps.update(&Event::Pon {
        actor: 1, target: 2, pai: t!(1p), consumed: [t!(1p), t!(1p)],
    }).unwrap();
    // Player 1 must discard; now has 4 melds + 2 closed tiles (after consuming 2 for pon)
    // Actually at this point player 1 has: 4 melds, hand should be very small
    ps.update(&Event::Dahai { actor: 1, pai: t!(2p), tsumogiri: false }).unwrap();

    // Get to a state where player 1 has a tsumo
    for actor in [3_u8, 0] {
        ps.update(&Event::Tsumo { actor, pai: t!(?) }).unwrap();
        ps.update(&Event::Dahai { actor, pai: t!(F), tsumogiri: true }).unwrap();
    }
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();

    // Patch with empty hand (simulating 4 melds + kans exhausting hand)
    let patch_tiles: &[crate::tile::Tile] = &[];
    ps.patch_hand(patch_tiles);

    // Verify tehai is all zeros
    assert!(ps.tehai.iter().all(|&c| c == 0), "tehai should be all zeros");
    // Shanten should be safe default (6)
    assert_eq!(ps.shanten, 6, "shanten should be 6 for empty hand");
    // tiles_seen should still include discards, melds, and dora indicators
    assert!(
        ps.tiles_seen.iter().sum::<u8>() > 0,
        "tiles_seen should not be all zeros even with empty hand"
    );
}

#[test]
fn patch_hand_full_hand_13_tiles() {
    // Patch with 13 tiles (3n+1 = 13 where n=4). Verify shanten and waits are calculated.
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(9s),
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 plays, player 1 draws
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(E), tsumogiri: true }).unwrap();

    // Player 1 draws and discards (back to 13 tiles)
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(N), tsumogiri: true }).unwrap();

    // Continue
    ps.update(&Event::Tsumo { actor: 2, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 2, pai: t!(S), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 3, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 3, pai: t!(W), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(P), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(F), tsumogiri: true }).unwrap();

    // Now player 1 has 13 tiles in hand after a draw-discard-draw cycle,
    // but we haven't given them a new tsumo yet. They have 13 tiles.
    // Actually after the last discard, they have 12+... let me trace:
    // Start: 13, tsumo: 14, discard N: 13, tsumo: 14, discard F: 13
    // So player 1 has 13 tiles right now (just discarded).
    // Wait - this means player 1 isn't the active player. Let me fix.
    // After player 1's discard, seat 2 would go. Let me just get
    // to a state where we know the hand size.

    // Player 2, 3, 0 play
    ps.update(&Event::Tsumo { actor: 2, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 2, pai: t!(C), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 3, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 3, pai: t!(C), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(F), tsumogiri: true }).unwrap();

    // Patch with tenpai hand: 1-9m, 1-3p, N (waiting on 4p)
    // This is 3n+1 = 13 tiles
    let patch_tiles = t![1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m, 9m, 1p, 2p, 3p, N];
    ps.patch_hand(&patch_tiles);

    // 13 tiles = 3n+1 (n=4), shanten and waits should be calculated
    assert_eq!(ps.tehai_len_div3, 4);
    // This hand: 123m 456m 789m 123p + N pair? No, N is single.
    // 123m, 456m, 789m = 3 mentsu, 123p = partial, N = single
    // Shanten = 0 (tenpai), waiting on 4p and N?
    // Actually: 123m, 456m, 789m, 12p + N -> waiting on 3p (for 123p mentsu, N pair)
    // Wait, we have 1p 2p 3p = mentsu, plus N = 1 tile left.
    // 123m 456m 789m 123p = 4 mentsu, N = tanki wait -> shanten = 0
    assert_eq!(ps.shanten, 0, "tenpai hand should have shanten 0");
    assert!(ps.waits[tuz!(N)], "should be waiting on N (tanki)");
}

#[test]
fn patch_hand_14_tiles() {
    // Patch with 14 tiles (3n+2 = 14 where n=4). Verify shanten
    // calculated but waits NOT updated (requires 3n+1).
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(9s),
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 plays, player 1 draws (now has 14 tiles)
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(E), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();

    // Patch with 14 tiles (3n+2)
    let patch_tiles = t![1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m, 9m, 1p, 2p, 3p, N, N];
    ps.patch_hand(&patch_tiles);

    // 14 tiles = 3*4 + 2, tehai_len_div3 = 4
    assert_eq!(ps.tehai_len_div3, 4);
    // Shanten should be calculated (for 3n+2 state).
    // update_shanten() clamps to max(0), so even a complete hand reads as 0.
    // Hand: 123m 456m 789m 123p NN -> complete hand, but clamped to 0
    assert_eq!(ps.shanten, 0, "3n+2 complete hand: shanten clamped to 0 by update_shanten");
    // Waits should NOT be updated for 3n+2 (patch_hand skips update_waits_and_furiten)
    // waits were reset to all false at the start of patch_hand
    assert!(
        ps.waits.iter().all(|&w| !w),
        "waits should not be populated for 3n+2 hand"
    );
}

#[test]
fn patch_hand_tiles_seen_with_kakan() {
    // Player 1 has a kakan (pon upgraded to kan).
    // fuuro_overview entry will have len=4 after kakan.
    // tiles_seen should count 3 from hand + 1 from kawa = 4 total.
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(9s),
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 discards 1m
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(1m), tsumogiri: true }).unwrap();

    // Player 1 pons 1m (consuming 2 from hand)
    ps.update(&Event::Pon {
        actor: 1,
        target: 0,
        pai: t!(1m),
        consumed: [t!(1m), t!(1m)],
    })
    .unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(E), tsumogiri: false }).unwrap();

    // Other players play
    for actor in [2_u8, 3, 0] {
        ps.update(&Event::Tsumo { actor, pai: t!(?) }).unwrap();
        ps.update(&Event::Dahai { actor, pai: t!(N), tsumogiri: true }).unwrap();
    }

    // Player 1 draws the 4th 1m and does kakan
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Kakan {
        actor: 1,
        pai: t!(1m),
        consumed: [t!(1m), t!(1m), t!(1m)],
    })
    .unwrap();
    ps.update(&Event::Dora { dora_marker: t!(2p) }).unwrap();

    // Rinshan tsumo and discard
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(W), tsumogiri: true }).unwrap();

    // Other players play
    for actor in [2_u8, 3, 0] {
        ps.update(&Event::Tsumo { actor, pai: t!(?) }).unwrap();
        ps.update(&Event::Dahai { actor, pai: t!(S), tsumogiri: true }).unwrap();
    }

    // Player 1 draws again
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();

    // Patch player 1's hand
    let patch_tiles = t![2m, 3m, 4p, 5p, 6p, 7s, 8s, 9s, P, P, C];
    ps.patch_hand(&patch_tiles);

    // The kakan meld has len=4 in fuuro_overview[0].
    // tiles_seen for 1m should be 4:
    //   3 from fuuro consumed (meld len=4 -> 3 from hand) + 1 from kawa (player 0's discard)
    assert_eq!(
        ps.tiles_seen[tuz!(1m)], 4,
        "kakan of 1m should result in tiles_seen[1m] = 4, got {}",
        ps.tiles_seen[tuz!(1m)]
    );
}

#[test]
fn patch_hand_tiles_seen_with_daiminkan() {
    // Player 1 has a daiminkan. fuuro_overview entry has len=4.
    // tiles_seen should count 3 from hand + 1 from kawa = 4 total.
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(9s),
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 discards 1m
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(1m), tsumogiri: true }).unwrap();

    // Player 1 daiminkan (3 from hand + 1 called)
    ps.update(&Event::Daiminkan {
        actor: 1,
        target: 0,
        pai: t!(1m),
        consumed: [t!(1m), t!(1m), t!(1m)],
    })
    .unwrap();
    ps.update(&Event::Dora { dora_marker: t!(2p) }).unwrap();

    // Rinshan tsumo and discard
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(E), tsumogiri: true }).unwrap();

    // Other players play
    for actor in [2_u8, 3, 0] {
        ps.update(&Event::Tsumo { actor, pai: t!(?) }).unwrap();
        ps.update(&Event::Dahai { actor, pai: t!(N), tsumogiri: true }).unwrap();
    }

    // Player 1 draws again
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();

    // Patch player 1's hand
    let patch_tiles = t![2m, 3m, 4p, 5p, 6p, 7s, 8s, 9s, P, P];
    ps.patch_hand(&patch_tiles);

    // The daiminkan meld has len=4 in fuuro_overview[0].
    // tiles_seen for 1m should be 4:
    //   3 from fuuro consumed (meld len=4 -> 3 from hand) + 1 from kawa (player 0's discard)
    assert_eq!(
        ps.tiles_seen[tuz!(1m)], 4,
        "daiminkan of 1m should result in tiles_seen[1m] = 4, got {}",
        ps.tiles_seen[tuz!(1m)]
    );
}

#[test]
fn patch_hand_akas_seen_ankan_5m() {
    // Player 0 has an ankan of 5m. After patch on player 1,
    // akas_seen[0] should be true and doras_seen should include the aka bonus.
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(9s),
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 draws and declares ankan of 5m
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Ankan {
        actor: 0,
        consumed: [t!(5m), t!(5m), t!(5m), t!(5mr)],
    })
    .unwrap();
    ps.update(&Event::Dora { dora_marker: t!(2p) }).unwrap();

    // Rinshan tsumo and discard
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(E), tsumogiri: true }).unwrap();

    // Other players play
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(W), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 2, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 2, pai: t!(N), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 3, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 3, pai: t!(F), tsumogiri: true }).unwrap();

    // Player 0 draws again
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(S), tsumogiri: true }).unwrap();

    // Player 1 draws
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();

    // Patch player 1's hand
    let patch_tiles = t![2m, 3m, 4p, 5p, 6p, 7s, 8s, 9s, P, P, C, C];
    ps.patch_hand(&patch_tiles);

    // ankan_overview stores deaka'd tiles, but patch_hand now sets akas_seen
    // for ankans of 5m/5p/5s since they necessarily contain the aka.
    assert!(
        ps.akas_seen[0],
        "ankan of 5m should set akas_seen[0] = true"
    );

    // tiles_seen for 5m should be 4
    assert_eq!(
        ps.tiles_seen[tuz!(5m)], 4,
        "ankan of 5m should contribute 4 to tiles_seen"
    );

    // doras_seen should include the +1 aka bonus from akas_seen[0]
    // dora_factor for 5m is 0 (dora indicator is 9s, so dora is 1s)
    // dora_factor for 2p+1=3p is 0 (second dora indicator is 2p, so dora is 3p)
    // The only "dora" is the aka bonus.
    // Check that the aka 5mr is reflected in doras_seen:
    // doras_seen = sum(tiles_seen[i] * dora_factor[i]) + count(akas_seen == true)
    // akas_seen[0] is true -> +1
    assert!(
        ps.doras_seen >= 1,
        "doras_seen should include at least 1 for the aka in ankan of 5m, got {}",
        ps.doras_seen
    );
}

#[test]
fn patch_hand_doras_owned_ankan_5m() {
    // Player 1 (self, seat 0 relative) has an ankan of 5m.
    // Dora indicator is 4m, making 5m the dora.
    // After patch, doras_owned[0] should include 4 (dora factor) + 1 (aka) = 5.
    let mut ps = PlayerState::new(1);
    ps.replay_mode = true;

    ps.update(&Event::StartKyoku {
        bakaze: t!(E),
        kyoku: 1,
        honba: 0,
        kyotaku: 0,
        oya: 0,
        scores: [25000; 4],
        dora_marker: t!(4m),   // dora indicator 4m -> dora is 5m
        tehais: [
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    })
    .unwrap();

    // Player 0 plays
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(E), tsumogiri: true }).unwrap();

    // Player 1 draws and declares ankan of 5m
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Ankan {
        actor: 1,
        consumed: [t!(5m), t!(5m), t!(5m), t!(5mr)],
    })
    .unwrap();
    ps.update(&Event::Dora { dora_marker: t!(9s) }).unwrap();

    // Rinshan tsumo and discard
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 1, pai: t!(W), tsumogiri: true }).unwrap();

    // Other players play
    ps.update(&Event::Tsumo { actor: 2, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 2, pai: t!(N), tsumogiri: true }).unwrap();
    ps.update(&Event::Tsumo { actor: 3, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 3, pai: t!(F), tsumogiri: true }).unwrap();

    // Player 0 plays
    ps.update(&Event::Tsumo { actor: 0, pai: t!(?) }).unwrap();
    ps.update(&Event::Dahai { actor: 0, pai: t!(S), tsumogiri: true }).unwrap();

    // Player 1 draws
    ps.update(&Event::Tsumo { actor: 1, pai: t!(?) }).unwrap();

    // Patch player 1's hand (no dora tiles in closed hand)
    let patch_tiles = t![1p, 2p, 3p, 7s, 8s, 9s, E, E, S, S, N, N];
    ps.patch_hand(&patch_tiles);

    // doras_owned[0] should be:
    // - From closed hand: no dora tiles -> 0
    // - From ankan: dora_factor[5m] * 4 = 1 * 4 = 4
    // - From ankan aka bonus: +1 (5m ankan contains aka)
    // Total: 5
    assert_eq!(
        ps.doras_owned[0], 5,
        "doras_owned[0] should be 5 (4 dora + 1 aka) for ankan of 5m with 4m indicator, got {}",
        ps.doras_owned[0]
    );
}
